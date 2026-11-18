#!/usr/bin/env python3
"""Parse one PDF or image with Infinity Parser and save JSON and Markdown.

By default the file goes to /v1/parse, whose stream reports each page as it
finishes. --endpoint chat sends it to /v1/chat/completions instead; both
endpoints save the same outputs.

Exit codes: 0 every page parsed; 1 the request failed (nothing written);
2 usage error, including an unsupported or oversized input, existing outputs
without --overwrite, or an output location that can't be written (nothing
sent); 3 outputs saved, but some pages failed (see failed_pages); 4 the parse
finished, but the outputs couldn't be saved in the output directory (stderr
says where a copy was saved instead).
"""

import argparse
import http.client
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# The helper modules next to this script; python3 -I or -P doesn't look there itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from file_utils import check_writable, env_file_value, inspect_input, save, save_elsewhere
from http_utils import OPENER, Body, error_detail, retry_hint, sse_events, url_problem

ENDPOINTS = {
    "parse": "/v1/parse",
    "chat": "/v1/chat/completions",
}
# The stream sends a heartbeat after 10 s without another event, so a longer
# silence means the connection is dead. The chat endpoint says nothing until
# the whole document is parsed.
STREAM_READ_TIMEOUT_S = 120
RESULT_TIMEOUT_S = 1800
# The service's limit on the file. The chat endpoint refuses a larger request
# by closing the connection mid-upload, so the client never sees its HTTP 413.
MAX_FILE_BYTES = 50 * 1024 * 1024
# A gateway can refuse a request, say for its key, before reading the whole
# upload, and close the connection on the rest of it, or without a response if
# the rest fit in the socket buffers.
REFUSED_HINT = ("; the service may have refused the request before reading all of it, "
                "as it does for a missing or invalid API key")


def config():
    return (
        os.environ.get("INFINITY_PARSER_SERVICE_URL") or env_file_value("INFINITY_PARSER_SERVICE_URL")
        or "https://parser-gw.openapi-inspire.inf.tech",
        os.environ.get("INFINITY_PARSER_API_KEY") or env_file_value("INFINITY_PARSER_API_KEY")
        or os.environ.get("INF_API_KEY") or env_file_value("INF_API_KEY"),
    )


def input_part(data_url, mime):
    if mime != "application/pdf":
        return {"type": "image_url", "image_url": {"url": data_url}}
    # The service requires a .pdf name; the local file name is not sent.
    return {"type": "file", "file": {"filename": "document.pdf", "file_data": data_url}}


def chat_request(data_url, mime, args):
    options = {
        "keep_header_footer": args.keep_header_footer == "true",
        "parse_chart": args.parse_chart == "true",
        "include_page_markers": args.include_page_markers == "true",
        "lang": args.lang,
    }
    if args.pages:
        options["pages"] = args.pages
    return {
        "model": f"infinity-parser-{args.tier}",
        "messages": [{"role": "user", "content": [input_part(data_url, mime)]}],
        "parser_options": options,
    }


# Stands in for the data URL while the chat request is encoded.
DATA_URL_SLOT = "data-url"


def chat_frame(mime, args):
    """Encode the chat request around the document's Base64 data URL: the JSON
    before the Base64, and after it. The Base64 is spliced in as it is sent:
    passing it through json.dumps would hold the document in memory several
    times over."""
    head, tail = json.dumps(chat_request(DATA_URL_SLOT, mime, args)).encode().split(
        json.dumps(DATA_URL_SLOT).encode(), 1)
    return head + f'"data:{mime};base64,'.encode(), b'"' + tail


def form_fields(args):
    """The options as /v1/parse form fields, where booleans are "true" or "false"."""
    fields = {
        # The default, but explicit: with "false" the service answers with JSON instead.
        "stream": "true",
        "tier": args.tier,
        "lang": args.lang,
        "parse_chart": args.parse_chart,
        "keep_header_footer": args.keep_header_footer,
        "include_page_markers": args.include_page_markers,
    }
    if args.pages:
        fields["pages"] = args.pages
    return fields


def multipart(boundary, fields, mime):
    """Encode the form fields and the document as the `file` part: the form
    before the document's content, and after it."""
    head = "".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        for name, value in fields.items()
    )
    # The service reads the type from the content; the local file name is not sent.
    filename = "document." + mime.split("/")[1]
    head += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f"Content-Type: {mime}\r\n\r\n")
    return head.encode(), f"\r\n--{boundary}--\r\n".encode()


def build_request(url, key, endpoint, path, size, mime, args):
    headers = {"Authorization": f"Bearer {key}"}
    if endpoint == "chat":
        head, tail = chat_frame(mime, args)
        data = Body(head, path, size, tail, b64=True)
        headers["Content-Type"] = "application/json"
    else:
        boundary = "----inftech" + os.urandom(12).hex()
        head, tail = multipart(boundary, form_fields(args), mime)
        data = Body(head, path, size, tail)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    # Without it urllib would send the body with chunked transfer encoding.
    headers["Content-Length"] = str(data.length)
    return urllib.request.Request(f"{url.rstrip('/')}{ENDPOINTS[endpoint]}", data=data, headers=headers)


class ParseFailed(Exception):
    """The service answered, but without a parse result."""


def checked(result):
    if not isinstance(result, dict) or not isinstance(result.get("md"), str):
        raise ParseFailed("Unexpected parse response: not a parse result with md")
    return result


def read_chat(response):
    try:
        completion = json.load(response)
        markdown = completion["choices"][0]["message"]["content"]
        result = completion["parser_result"]
    except (ValueError, LookupError, TypeError):
        raise ParseFailed("Unexpected parse response: not a chat completion with parser_result") from None
    if not isinstance(markdown, str) or not isinstance(result, dict):
        raise ParseFailed("Unexpected parse response: missing markdown or parser_result")
    return dict(result, md=markdown)


def read_stream(response):
    """Report progress on stderr and return the result from the `done` event."""
    for event, data in sse_events(response):
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if event == "done":
            return checked(payload)
        if event == "failed":
            raise ParseFailed(
                f"Parse failed (status {payload.get('status_code')}, {payload.get('error_type')}): "
                f"{payload.get('error')}{retry_hint(payload.get('retry_after'))}")
        if event == "progress":
            # nano parses the whole document before it can report any page.
            later = "" if payload.get("page_streaming", True) else "; pages are reported once all are parsed"
            print(f"Parsing {payload.get('selected_page_count')} of {payload.get('total_pages')} pages{later}",
                  file=sys.stderr)
        elif event in ("page_parsed", "page_failed"):
            outcome = "parsed" if event == "page_parsed" else f"failed: {payload.get('error')}"
            print(f"Page {payload.get('page')} {outcome} "
                  f"({payload.get('completed_pages')}/{payload.get('selected_page_count')})", file=sys.stderr)
    raise ParseFailed("Request failed: the stream ended before the parse finished or failed")


READERS = {"parse": read_stream, "chat": read_chat}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="PDF or single-frame image")
    parser.add_argument("-o", "--output", type=Path, help="output directory (default: next to input)")
    parser.add_argument("-t", "--tier", choices=("nano", "flash", "pro"), default="flash", help="parse tier (default: flash)")
    parser.add_argument("--endpoint", choices=tuple(ENDPOINTS), default="parse",
                        help="parse (/v1/parse, per-page progress) or chat (/v1/chat/completions) "
                             "(default: parse)")
    parser.add_argument("--pages", help="1-based pages, e.g. 1-3,5; an image is page 1")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh", help="language of figure descriptions (default: zh)")
    parser.add_argument("--keep-header-footer", choices=("true", "false"), default="false")
    parser.add_argument("--parse-chart", choices=("true", "false"), default="true")
    parser.add_argument("--include-page-markers", choices=("true", "false"), default="false",
                        help="mark where each page starts in the Markdown (default: false)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args()
    # Paths and results can hold Chinese, which the locale's encoding may not
    # (cp1252 on a Windows pipe), and an unencodable print would fail the run.
    # A file name that isn't valid UTF-8 can't be encoded even so; escape it.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    try:
        if not args.file.is_file():
            parser.error(f"file not found: {args.file}")
        size, mime = inspect_input(args.file)
    except OSError as exc:
        parser.error(f"cannot read {args.file}: {exc.strerror or exc}")
    if mime is None:
        parser.error(f"unsupported input: {args.file}; use a PDF or a PNG, JPEG, WEBP, BMP, TIFF or GIF image")
    if size > MAX_FILE_BYTES:
        parser.error(f"{args.file} is larger than the service's {MAX_FILE_BYTES // 2**20} MiB limit")
    url, key = config()
    if not key:
        parser.error("an API key is required: set INFINITY_PARSER_API_KEY or INF_API_KEY "
                     "in ~/.config/inftech/.env or the environment")
    problem = url_problem(url)
    if problem:
        parser.error(f"INFINITY_PARSER_SERVICE_URL {problem}: {url!r}")
    if " " in key or not key.isprintable() or not key.isascii():
        parser.error("the API key contains spaces, line breaks or other characters a key can't have; "
                     "check INFINITY_PARSER_API_KEY or INF_API_KEY")
    output = args.output or args.file.parent
    json_path = output / f"{args.file.stem}.json"
    md_path = output / f"{args.file.stem}.md"
    try:
        if any(path.exists() and path.samefile(args.file) for path in (json_path, md_path)):
            parser.error(f"the output would replace the input {args.file}; choose another directory with --output")
        if not args.overwrite and (os.path.lexists(json_path) or os.path.lexists(md_path)):
            parser.error(f"output exists: {json_path} or {md_path}; use --overwrite")
        check_writable(output, (json_path, md_path))
    except OSError as exc:
        parser.error(f"cannot save the outputs in {output}: {exc}")

    request = build_request(url, key, args.endpoint, args.file, size, mime, args)
    stream = args.endpoint == "parse"
    try:
        with OPENER.open(request, timeout=STREAM_READ_TIMEOUT_S if stream else RESULT_TIMEOUT_S) as response:
            result = READERS[args.endpoint](response)
    except urllib.error.HTTPError as exc:
        retry = exc.headers.get("Retry-After") if exc.headers else None
        detail = error_detail(exc)
        if exc.code in (401, 403) and "record not found" in detail:
            # The platform gateway took the key but has no service at this address.
            detail += "; no service is deployed at this URL: check INFINITY_PARSER_SERVICE_URL"
        parser.exit(1, f"HTTP {exc.code}: {detail}{retry_hint(retry)}\n")
    except urllib.error.URLError as exc:
        if request.data.started and isinstance(exc.reason, OSError):
            refused = REFUSED_HINT if isinstance(exc.reason, ConnectionError) else ""
            parser.exit(1, f"Request failed: the upload stopped: {exc.reason}{refused}\n")
        parser.exit(1, f"Request failed: {exc.reason}\n")
    except TimeoutError:
        if stream:
            parser.exit(1, f"Request failed: the service sent nothing for {STREAM_READ_TIMEOUT_S} s\n")
        parser.exit(1, "Request failed: timed out waiting for the parse result\n")
    except (OSError, http.client.HTTPException) as exc:
        # The service or a gateway closed the connection before the result arrived.
        refused = REFUSED_HINT if isinstance(exc, http.client.RemoteDisconnected) and request.data.started else ""
        parser.exit(1, f"Request failed: connection lost: {exc!r}{refused}\n")
    except ParseFailed as exc:
        parser.exit(1, f"{exc}\n")

    outputs = [(json_path, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()),
               (md_path, result["md"].encode())]
    failed = f"Failed pages: {result['failed_pages']}\n" if result.get("failed_pages") else ""
    try:
        save(outputs, args.overwrite)
    except OSError as exc:
        if isinstance(exc, FileExistsError):
            problem = f"{exc.filename} was saved by something else during the parse, perhaps another run"
        else:
            problem = f"cannot save the outputs in {output}: {exc}"
        try:
            kept = f"the outputs are saved in {save_elsewhere(outputs)} instead"
        except OSError as rescue_exc:
            kept = f"they can't be saved in a temporary directory either: {rescue_exc}"
        parser.exit(4, f"Parsed, but {problem}; {kept}\n{failed}")
    print(f"Saved {json_path} and {md_path}")
    if failed:
        parser.exit(3, failed)


if __name__ == "__main__":
    main()
