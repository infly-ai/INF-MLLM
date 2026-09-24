# Parse API response contract

## Synchronous `/v1/parse`

The POST request accepts optional `tier` (`nano`, `flash`, or `pro`; default `flash`) and waits for the complete result. A partial PDF failure still returns HTTP 200, so inspect `failed_pages`. Unselected pages never appear there. Blocks keep original PDF page numbers and use bounding boxes normalized to 0–1. The response does not include preview page dimensions; retain platform preview metadata separately. Image/chart description failure is logged but does not enter `failed_pages`. For Flash and Pro, `engine` is `inf_parser`.

| Status | Meaning |
| --- | --- |
| 400 | Empty, invalid, or unsupported input; multi-frame image; incompatible tier/input; invalid page selection |
| 413 | File exceeds size limit |
| 422 | Missing required `file` or invalid field type |
| 502 | Upstream failure, including every selected PDF page failing |

Business errors use `{"error":{"message":"...","type":"...","param":null,"code":null}}`. FastAPI returns `detail` for 422 form validation errors.

## Streaming `/v1/parse/stream`

Use POST with the same multipart fields; omitted `tier` defaults to `flash`. The response is SSE (`text/event-stream`), and the response header and every event contain `request_id`. Events end with a blank line; each `data:` line contains one JSON object with text newlines escaped.

| Event | Data |
| --- | --- |
| `start` | `total_pages`, `selected_pages`, `selected_page_count`, `completed_pages=0`, `progress=0`, `page_streaming` |
| `page_parsed` | `page`, that page's `blocks` and `markdown`, `total_pages`, `selected_page_count`, `completed_pages`, `progress`, `blocks_so_far` |
| `page_failed` | `page`, string `error`, and the same count/progress fields as `page_parsed` |
| `heartbeat` | `elapsed_sec` every 10 seconds without a page event; connection liveness only |
| `done` | Complete `/v1/parse` result plus `request_id` |
| `failed` | Terminal `error`, `error_type`, and `status_code`; no `done` follows |

Page events arrive in completion order, which can differ from page order. `progress` uses the selected page count and includes failed pages. The final `done.blocks` are sorted by page and reading order, with continuous `order` values. Associate provisional and final blocks by stable block `id`, and replace the cached result with `done`.

Input, size, compatibility, PDF structure, and page errors return JSON before SSE starts. After SSE starts, HTTP status is already 200: partial page failures end with `done`; every selected page failing ends with `failed`. Treat a disconnected stream without a terminal `done` or `failed` as incomplete.
