#!/usr/bin/env python3
"""Parse one PDF or image with Infinity Parser and save JSON and Markdown."""

import argparse
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path


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


def multipart(path, fields):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="upload.bin"\r\n'
        'Content-Type: application/octet-stream\r\n\r\n'.encode()
    )
    # ponytail: buffers the upload in memory; stream it if Gateway file limits grow substantially.
    parts.extend((path.read_bytes(), b"\r\n", f"--{boundary}--\r\n".encode()))
    return boundary, b"".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="PDF or single-frame image")
    parser.add_argument("-o", "--output", type=Path, help="output directory (default: next to input)")
    parser.add_argument("-t", "--tier", choices=("nano", "flash", "pro"), default="flash", help="parse tier (default: flash)")
    parser.add_argument("--pages", help="1-based PDF pages, e.g. 1-3,5")
    parser.add_argument("--keep-header-footer", choices=("true", "false"), default="false")
    parser.add_argument("--parse-chart", choices=("true", "false"), default="true")
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"file not found: {args.file}")
    url, key = config()
    if not url or not key:
        parser.error("INF_API_URL and INF_API_KEY are required in the skill .env or environment")

    fields = {
        "tier": args.tier,
        "parse_chart": args.parse_chart,
        "keep_header_footer": args.keep_header_footer,
    }
    if args.pages:
        fields["pages"] = args.pages
    boundary, body = multipart(args.file, fields)
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/parse",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("error", {}).get("message", exc.reason)
        except (ValueError, AttributeError):
            detail = exc.reason
        parser.exit(1, f"HTTP {exc.code}: {detail}\n")
    except urllib.error.URLError as exc:
        parser.exit(1, f"Request failed: {exc.reason}\n")

    try:
        result = json.loads(raw)
    except ValueError:
        parser.exit(1, "Unexpected parse response: invalid JSON\n")
    if not isinstance(result, dict) or not isinstance(result.get("markdown"), str):
        parser.exit(1, "Unexpected parse response: missing markdown\n")
    output = args.output or args.file.parent
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{args.file.stem}.json"
    md_path = output / f"{args.file.stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(result["markdown"], encoding="utf-8")
    print(f"Saved {json_path} and {md_path}")
    if result.get("failed_pages"):
        parser.exit(2, f"Failed pages: {result['failed_pages']}\n")


if __name__ == "__main__":
    main()
