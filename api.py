"""
REST API service for dots.ocr - Document Layout Parsing & OCR.

Usage:
    python api.py
    python api.py --port 5000 --host 0.0.0.0
    python api.py --vllm-ip 192.168.1.10 --vllm-port 8000

Endpoints:
    POST /parse           - Full layout detection + OCR (layout + content)
    POST /layout          - Layout detection only (bboxes + categories, no text)
    POST /ocr             - Plain text extraction (no layout info)
    POST /grounding-ocr   - Extract text from a specific bounding box region
    GET  /health          - Health check
"""

import argparse
import base64
import json
import os
import tempfile
import uuid
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from dots_ocr.parser import DotsOCRParser

app = FastAPI(
    title="dots.ocr API",
    description="REST API for document layout parsing and OCR using dots.ocr",
    version="1.0.0",
)

parser: DotsOCRParser = None


# ---------- Response models ----------

class LayoutElement(BaseModel):
    bbox: List[int]
    category: str
    text: Optional[str] = None


class PageResult(BaseModel):
    page_no: int
    layout: Optional[List[LayoutElement]] = None
    markdown: Optional[str] = None
    markdown_no_hf: Optional[str] = None
    layout_image_base64: Optional[str] = None


class ParseResponse(BaseModel):
    pages: List[PageResult]


class OCRResponse(BaseModel):
    pages: List[dict]


# ---------- Helpers ----------

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
    ext = os.path.splitext(upload.filename or "file")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    tmp_dir = tempfile.mkdtemp()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(tmp_dir, safe_name)
    content = upload.file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    return tmp_path


def _read_file_text(path: str) -> Optional[str]:
    """Read a text file if it exists, else return None."""
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _read_image_base64(path: str) -> Optional[str]:
    """Read an image file and return base64-encoded string."""
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


def _build_page_result(result: dict, include_layout_image: bool = False) -> PageResult:
    """Build a PageResult from the raw parser result dict."""
    page_no = result.get("page_no", 0)

    # Read layout JSON
    layout = None
    layout_path = result.get("layout_info_path")
    if layout_path and os.path.isfile(layout_path):
        with open(layout_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            layout = []
            for item in raw:
                if isinstance(item, dict) and "bbox" in item:
                    layout.append(LayoutElement(
                        bbox=item["bbox"],
                        category=item.get("category", "Unknown"),
                        text=item.get("text"),
                    ))

    # Read markdown content
    markdown = _read_file_text(result.get("md_content_path"))
    markdown_no_hf = _read_file_text(result.get("md_content_nohf_path"))

    # Optionally include layout visualization image
    layout_image_base64 = None
    if include_layout_image:
        layout_image_base64 = _read_image_base64(result.get("layout_image_path"))

    return PageResult(
        page_no=page_no,
        layout=layout,
        markdown=markdown,
        markdown_no_hf=markdown_no_hf,
        layout_image_base64=layout_image_base64,
    )


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
async def parse(
    file: UploadFile = File(...),
    include_layout_image: bool = Form(False),
    dpi: Optional[int] = Form(None),
    fitz_preprocess: bool = Form(True),
):
    """
    Full document parsing: layout detection + OCR content extraction.

    Returns layout elements (bbox, category, text) and markdown for each page.
    """
    tmp_path = _save_upload(file)
    try:
        output_dir = tempfile.mkdtemp()
        results = parser.parse_file(
            tmp_path,
            output_dir=output_dir,
            prompt_mode="prompt_layout_all_en",
            fitz_preprocess=fitz_preprocess,
        )
        pages = [_build_page_result(r, include_layout_image) for r in results]
        return ParseResponse(pages=pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/layout", response_model=ParseResponse)
async def layout(
    file: UploadFile = File(...),
    include_layout_image: bool = Form(False),
    fitz_preprocess: bool = Form(True),
):
    """
    Layout detection only: returns bounding boxes and categories (no text content).
    """
    tmp_path = _save_upload(file)
    try:
        output_dir = tempfile.mkdtemp()
        results = parser.parse_file(
            tmp_path,
            output_dir=output_dir,
            prompt_mode="prompt_layout_only_en",
            fitz_preprocess=fitz_preprocess,
        )
        pages = [_build_page_result(r, include_layout_image) for r in results]
        return ParseResponse(pages=pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    file: UploadFile = File(...),
):
    """
    Plain text extraction (OCR only, no layout information).
    """
    tmp_path = _save_upload(file)
    try:
        output_dir = tempfile.mkdtemp()
        results = parser.parse_file(
            tmp_path,
            output_dir=output_dir,
            prompt_mode="prompt_ocr",
        )
        pages = []
        for r in results:
            text = _read_file_text(r.get("md_content_path"))
            pages.append({
                "page_no": r.get("page_no", 0),
                "text": text,
            })
        return OCRResponse(pages=pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/grounding-ocr")
async def grounding_ocr(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
):
    """
    Extract text from a specific bounding box region [x1, y1, x2, y2] in the image.

    Only works with image files (not PDF).
    """
    tmp_path = _save_upload(file)
    try:
        ext = os.path.splitext(tmp_path)[1].lower()
        if ext == ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Grounding OCR is only supported for image files, not PDF.",
            )
        output_dir = tempfile.mkdtemp()
        results = parser.parse_file(
            tmp_path,
            output_dir=output_dir,
            prompt_mode="prompt_grounding_ocr",
            bbox=[x1, y1, x2, y2],
        )
        text = _read_file_text(results[0].get("md_content_path")) if results else None
        return {"bbox": [x1, y1, x2, y2], "text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ---------- Main ----------

def create_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="dots.ocr REST API server")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    ap.add_argument("--vllm-protocol", type=str, default="http", choices=["http", "https"])
    ap.add_argument("--vllm-ip", type=str, default="localhost", help="vLLM server IP")
    ap.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    ap.add_argument("--model-name", type=str, default="model", help="Model name in vLLM")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--num-threads", type=int, default=16, help="Thread pool size for PDF pages")
    ap.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    ap.add_argument("--use-hf", action="store_true", help="Use HuggingFace backend instead of vLLM")
    return ap


if __name__ == "__main__":
    import uvicorn

    args = create_parser().parse_args()

    parser = DotsOCRParser(
        protocol=args.vllm_protocol,
        ip=args.vllm_ip,
        port=args.vllm_port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_tokens,
        num_thread=args.num_threads,
        dpi=args.dpi,
        use_hf=args.use_hf,
    )

    uvicorn.run(app, host=args.host, port=args.port)
