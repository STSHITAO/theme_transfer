import base64
import io
import json
import mimetypes
import os
import shutil
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image


def generate_candidates(prompt, style_ref_paths, target_layout, case_id, root_dir=None, n=4, size="2K"):
    root = Path(root_dir) if root_dir else Path.cwd()
    load_dotenv(root / ".env")
    case_dir = root / "data" / "cases" / case_id
    output_dir = root / "data" / "outputs" / case_id / "candidates"
    case_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _mock_mode():
        response = {
            "mock": True,
            "message": "MOCK_MODE=true，未调用 Wan API。",
            "input_images": [*style_ref_paths[:3], target_layout],
        }
        response_path = _save_response(response, case_dir)
        return {
            "candidate_paths": _create_mock_candidates(target_layout, output_dir, n),
            "wan_response_path": response_path,
        }

    response = _call_wan(prompt, style_ref_paths[:3], target_layout, n=n, size=size)
    response_path = _save_response(_response_to_json(response), case_dir)
    urls = _extract_image_urls(response)
    if not urls:
        raise RuntimeError(f"Wan API returned no image URLs. Raw response saved to {response_path}")
    return {
        "candidate_paths": _download_candidates(urls[:n], output_dir),
        "wan_response_path": response_path,
    }


def _mock_mode():
    return os.getenv("MOCK_MODE", "false").lower() == "true"


def _create_mock_candidates(target_layout, output_dir, n):
    candidate_paths = []
    for index in range(1, n + 1):
        path = output_dir / f"candidate_{index:02d}.png"
        shutil.copyfile(target_layout, path)
        candidate_paths.append(str(path))
    return candidate_paths


def _call_wan(prompt, style_ref_paths, target_layout, n, size):
    base_url = os.getenv("ALI_IMAGE_BASE_URL")
    model = os.getenv("ALI_IMAGE_MODEL")
    api_key = os.getenv("ALI_IMAGE_API_KEY")
    if not base_url or not model or not api_key:
        raise RuntimeError("Missing ALI_IMAGE_BASE_URL, ALI_IMAGE_MODEL, or ALI_IMAGE_API_KEY")

    import dashscope
    from dashscope.aigc.image_generation import ImageGeneration
    from dashscope.api_entities.dashscope_response import Message

    dashscope.base_http_api_url = base_url
    message = Message(
        role="user",
        content=[
            {"text": prompt},
            *[{"image": _image_data_url(path)} for path in [*style_ref_paths, target_layout]],
        ],
    )
    return ImageGeneration.call(
        model=model,
        api_key=api_key,
        messages=[message],
        n=n,
        size=size,
        watermark=False,
    )


def _save_response(response, case_dir):
    path = case_dir / "wan_response.json"
    path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _response_to_json(response):
    if isinstance(response, dict):
        return response
    if hasattr(response, "__dict__"):
        return response.__dict__
    return {"raw_response": str(response)}


def _extract_image_urls(response):
    data = _response_to_json(response)
    urls = []
    for choice in data.get("output", {}).get("choices", []):
        message = choice.get("message", {})
        for content in message.get("content", []):
            if content.get("type") == "image" and content.get("image"):
                urls.append(content["image"])
    if urls:
        return urls

    for item in data.get("output", {}).get("results", []):
        url = item.get("url") or item.get("image_url")
        if url:
            urls.append(url)
    return urls


def _download_candidates(urls, output_dir):
    paths = []
    for index, url in enumerate(urls, start=1):
        output_path = output_dir / f"candidate_{index:02d}.png"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        paths.append(str(output_path))
    return paths


def _image_data_url(path, max_size=(1024, 1024), quality=88):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"
