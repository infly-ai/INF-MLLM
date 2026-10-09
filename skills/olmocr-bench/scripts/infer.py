#!/usr/bin/env python3
"""Parse olmOCR-Bench PDFs with Infinity Parser and write benchmark-ready Markdown."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from postprocess import postprocess

SKILL_DIR = Path(__file__).resolve().parent.parent


def config():
    values = {}
    env_file = SKILL_DIR / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return (
        os.environ.get("INF_API_URL") or values.get("INF_API_URL"),
        os.environ.get("INF_API_KEY") or values.get("INF_API_KEY"),
    )


def parse_pdf(url, key, path, tier, retries=3):
    # olmOCR-Bench scores page 1 only; headers/footers and figure descriptions must stay out of the Markdown.
    payload = {"tier": tier, "pages": "1", "keep_header_footer": "false", "parse_chart": "false"}
    pdf_bytes = path.read_bytes()
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                f"{url.rstrip('/')}/v1/parse",
                headers={"Authorization": f"Bearer {key}"},
                data=payload,
                files={"file": ("upload.pdf", pdf_bytes, "application/pdf")},
                timeout=1800,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("failed_pages"):
                raise RuntimeError(f"failed pages: {result['failed_pages']}")
            if not isinstance(result.get("markdown"), str):
                raise RuntimeError("response is missing markdown")
            return result
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            # 4xx errors (except rate limiting) will not succeed on retry.
            status = exc.response.status_code if isinstance(exc, requests.HTTPError) else None
            if status is not None and status < 500 and status != 429:
                raise RuntimeError(f"HTTP {status}: {exc.response.reason}") from exc
            if attempt == retries:
                raise RuntimeError(str(exc)) from exc
            time.sleep(5 * attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True, help="olmOCR-bench/bench_data/pdfs")
    parser.add_argument("--output-dir", type=Path, required=True, help="candidate folder inside bench_data")
    parser.add_argument("-t", "--tier", choices=("nano", "flash", "pro"), default="flash", help="parse tier (default: flash)")
    parser.add_argument("-j", "--workers", type=int, default=8, help="concurrent requests (default: 8)")
    args = parser.parse_args()

    url, key = config()
    if not url or not key:
        parser.error("INF_API_URL and INF_API_KEY are required in the skill .env or environment")

    pdfs = sorted(args.pdf_dir.rglob("*.pdf"))
    if not pdfs:
        parser.error(f"no PDFs found under {args.pdf_dir}")
    raw_dir = args.output_dir / "raw"

    def raw_path(pdf):
        return raw_dir / pdf.relative_to(args.pdf_dir).with_suffix(".json")

    pending = [pdf for pdf in pdfs if not raw_path(pdf).exists()]
    print(f"{len(pdfs)} PDFs, {len(pdfs) - len(pending)} already parsed, {len(pending)} to parse (tier={args.tier})")

    def run(pdf):
        result = parse_pdf(url, key, pdf, args.tier)
        out = raw_path(pdf)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, pdf): pdf for pdf in pending}
        for i, future in enumerate(as_completed(futures), 1):
            pdf = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed.append(pdf)
                print(f"[{i}/{len(pending)}] FAILED {pdf.relative_to(args.pdf_dir)}: {exc}", file=sys.stderr)
            else:
                print(f"[{i}/{len(pending)}] {pdf.relative_to(args.pdf_dir)}")

    # Always rebuild the Markdown from raw responses so post-processing changes apply to every file.
    written = 0
    for pdf in pdfs:
        if not raw_path(pdf).exists():
            continue
        markdown = json.loads(raw_path(pdf).read_text(encoding="utf-8"))["markdown"]
        category = pdf.parent.name
        md = args.output_dir / category / f"{pdf.stem}_pg1_repeat1.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(postprocess(markdown, category), encoding="utf-8")
        written += 1

    print(f"Wrote {written}/{len(pdfs)} Markdown files to {args.output_dir}")
    if failed:
        sys.exit(f"{len(failed)} PDFs failed; re-run the same command to retry them before scoring.")


if __name__ == "__main__":
    main()
