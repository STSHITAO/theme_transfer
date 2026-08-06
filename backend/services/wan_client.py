import base64
import io
import json
import mimetypes
import os
import shutil
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image


class WanApiError(RuntimeError):
    """A structured error returned by Wan instead of a usable image payload."""

    def __init__(self, message, *, status_code=None, code=None, response_path=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.response_path = response_path


def generate_candidates(
    prompt,
    style_ref_paths,
    target_layout,
    case_id,
    root_dir=None,
    n=3,
    size="2K",
    case_dir=None,
    output_dir=None,
):
    root = Path(root_dir) if root_dir else Path.cwd()
    load_dotenv(root / ".env")
    case_dir = Path(case_dir) if case_dir else root / "data" / "cases" / case_id
    output_dir = Path(output_dir) if output_dir else root / "data" / "outputs" / case_id / "candidates"
    case_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_candidate_outputs(output_dir)

    if _mock_mode():
        response = {
            "mock": True,
            "message": "MOCK_MODE=true，未调用 Wan API。",
            "input_images": [target_layout, *style_ref_paths[:3]],
        }
        response_path = _save_response(response, case_dir)
        return {
            "candidate_paths": _create_mock_candidates(target_layout, output_dir, n),
            "wan_response_path": response_path,
        }

    response = _call_wan(prompt, style_ref_paths[:3], target_layout, n=n, size=size)
    response_path = _save_response(_response_to_json(response), case_dir)
    try:
        urls = _extract_image_urls(response)
    except WanApiError as exc:
        exc.response_path = response_path
        raise
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


def _clear_candidate_outputs(output_dir):
    for path in output_dir.glob("candidate_*.png"):
        if path.is_file():
            path.unlink()


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
    role_lines = [
        "[INPUT IMAGE ROLE MAP]",
        (
            "IMAGE_1 = TARGET_IMAGE — ONLY IDENTITY SOURCE: the output subject, logo, text, symbol, silhouette, "
            "object category, geometry, and layout must come only from this image."
        ),
        *[
            (
                f"IMAGE_{index + 1} = STYLE_REFERENCE_{index}: learn only visual treatment; do not copy its subject, "
                "logo, text, symbol, silhouette, object category, geometry, or layout."
            )
            for index, _ in enumerate(style_ref_paths, start=1)
        ],
        (
            "[FINAL IDENTITY CHECK] Generate the TARGET_IMAGE identity with the shared visual treatment. "
            "Any identity copied from a STYLE_REFERENCE makes the output invalid."
        ),
    ]
    role_prompt = prompt.rstrip() + "\n\n" + "\n".join(role_lines)
    content = [
        {"text": role_prompt},
        *[{"image": _image_data_url(path)} for path in [target_layout, *style_ref_paths]],
    ]
    message = Message(role="user", content=content)
    last_error = None
    for _ in range(2):
        try:
            return ImageGeneration.call(
                model=model,
                api_key=api_key,
                messages=[message],
                n=n,
                size=size,
                watermark=False,
            )
        except requests.exceptions.RequestException as exc:
            last_error = exc
    raise last_error


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
    output = data.get("output") or {}
    if not isinstance(output, dict):
        output = {}
    urls = []
    for choice in output.get("choices", []):
        message = choice.get("message", {})
        for content in message.get("content", []):
            if content.get("type") == "image" and content.get("image"):
                urls.append(content["image"])
    if urls:
        return urls

    for item in output.get("results", []):
        url = item.get("url") or item.get("image_url")
        if url:
            urls.append(url)
    if urls:
        return urls

    status_code = data.get("status_code")
    code = data.get("code")
    message = data.get("message")
    if status_code or code or message:
        details = ", ".join(
            str(value)
            for value in (f"status_code={status_code}" if status_code else None, code, message)
            if value
        )
        raise WanApiError(f"Wan API rejected the image request: {details}", status_code=status_code, code=code)
    return urls


def _download_candidates(urls, output_dir, max_attempts=3):
    paths = []
    for index, url in enumerate(urls, start=1):
        output_path = output_dir / f"candidate_{index:02d}.png"
        temporary_path = output_path.with_suffix(output_path.suffix + ".part")
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                temporary_path.write_bytes(response.content)
                temporary_path.replace(output_path)
                last_error = None
                break
            except requests.exceptions.RequestException as exc:
                last_error = exc
                temporary_path.unlink(missing_ok=True)
                if attempt < max_attempts:
                    time.sleep(attempt)
        if last_error is not None:
            raise RuntimeError(
                f"Wan candidate download failed after {max_attempts} attempts for candidate {index}."
            ) from last_error
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
