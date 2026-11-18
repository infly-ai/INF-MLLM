# Parse API contract

The service has two parse endpoints. They accept the same inputs and options and produce the same parse result; they differ in how the file is sent and how the result comes back.

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /v1/parse` | `multipart/form-data` upload | server-sent events per page, ending with the parse result; with `stream=false`, the parse result as JSON once parsing finishes |
| `POST /v1/chat/completions` | OpenAI Chat Completions JSON with a Base64 file | a chat completion whose message is the Markdown |

Every endpoint requires `Authorization: Bearer <API_KEY>`. The bundled `scripts/parse.py` streams `/v1/parse` by default; `--endpoint chat` uses `/v1/chat/completions`. Whichever it uses, it saves the parse result as `<stem>.json` and its `md` as `<stem>.md`.

## Inputs and options

A request carries one PDF or one PNG, JPEG, WEBP, BMP, TIFF or GIF image of at most 50 MiB. The service checks the decoded content rather than the filename or MIME type, and rejects multi-frame images. An image is page 1. A request can select at most 1000 pages; leave out `pages` on a longer document and the service rejects it, so select the pages in batches.

The tier is `nano`, `flash` or `pro`. `/v1/parse` takes it as the `tier` field, which defaults to `flash`, and rejects a `model` field. The chat endpoint takes it as `model`: `infinity-parser-nano`, `infinity-parser-flash` or `infinity-parser-pro`.

The remaining options are optional. `/v1/parse` takes them as form fields, with booleans written as `true` or `false`. The chat endpoint takes them in a top-level `parser_options` object, with JSON booleans.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pages` | string | all pages | 1-based pages, e.g. `1-3,5`; an image has only page 1 |
| `lang` | `zh` or `en` | `zh` | language of figure and chart descriptions |
| `parse_chart` | boolean | `true` | describe charts and figures |
| `keep_header_footer` | boolean | `false` | keep headers, footers and page footnotes in the Markdown; their nodes are always kept |
| `include_page_markers` | boolean | `false` | insert `%%% Page N` lines into the Markdown |

`/v1/parse` also takes `stream`, default `true`, which picks server-sent events; `false` returns the parse result as JSON. The chat endpoint has no `stream` option.

## Parse result

The parse result has `schema_version` (3), `model`, `total_pages`, `parsed_pages`, `failed_pages`, `json`, `md`, `request_id`, `usage` and `warnings`.

- `total_pages` counts the whole document, not the selection; an image has 1.
- `parsed_pages` is a string of the parsed pages with runs merged, such as `"1-3,5-6"`, `"1"`, or `""` when none parsed. `failed_pages` is a list of page numbers. A partial failure still returns HTTP 200, so inspect `failed_pages`; unselected pages never appear there.
- `json` is the document in the parsedoc 2.0 format with `type` `"parsedoc"`, `version` `"2.0"`, `meta`, `data.nodes` and `logical_doctree`. The document format version is independent of `schema_version`.
- `md` is the Markdown.
- `usage.pages_processed` counts the parsed pages. The token counts in `usage` are `null` for `nano`, and `usage_complete` says whether they cover every model call. `usage.prompt_tokens_details.cached_tokens` is the part of `prompt_tokens` served from cache; `cache_usage_complete` says whether that count is complete.
- Image or chart description failures do not enter `failed_pages`; they are reported in `warnings`, which is empty when there are none.

`json.meta` describes the uploaded file: `raw_type`, `raw_name` (the uploaded filename), `raw_md5`, `raw_size` (bytes, as a string), `page_width` and `page_height` (the first page of the file, in points, even if that page wasn't selected), and `attrs`, a JSON object serialized as a string, such as `"{\"lang\": null}"`. It can also have `title`.

`json.data.nodes` lists the nodes in reading order. Each node has:

| Field | Meaning |
| --- | --- |
| `id` | unique in the document, `p<page>-n<n>` with `n` from 1, such as `p56-n1`; not stable between requests |
| `parent_id` | ID of the parent node, such as a section title, or `null` for a top-level node, header or footer |
| `level` | only on titles, when the level is known |
| `type` | `apara` for text, `atable` for a table, `afigure` for an image |
| `category` | what the node is, such as `title`, `text`, `table`, `table_caption`, `header` or `footer` |
| `content`, `ct` | text as Markdown with `ct` `md`, or a table as HTML with `ct` `html`, keeping `rowspan` and `colspan`; an image has `summary` instead of both |
| `refer.locs` | every place the node appears, one `{"page", "bbox"}` each; a node across columns or pages has several. `page` is the page number in the original file, and `bbox` is `[x0, y0, x1, y1]` normalized to 0–1 within that page, or `null` when unknown |
| `size` | font size for `nano` when known, otherwise `null` |

### Document structure

`json.logical_doctree` groups nodes by their `parent_id` links. Each node in `json.data.nodes` appears once in the tree except headers and footers. Look up each ID in `json.data.nodes` for its content. Siblings follow array order. For the reading order of the whole document, use `json.data.nodes`.

For example, a section title `p1-n1` and a paragraph whose `parent_id` is `p1-n1` produce this tree:

```json
{
  "id": null,
  "childs": [
    {"id": "p1-n1", "childs": [{"id": "p1-n2"}]}
  ]
}
```

The root always has a `childs` array, empty when the tree has no nodes. Leaves contain only `id`.

## `POST /v1/parse` with `stream=false`

```bash
curl "$INFINITY_PARSER_SERVICE_URL/v1/parse" \
  -H "Authorization: Bearer $INFINITY_PARSER_API_KEY" \
  -F file=@report.pdf \
  -F stream=false \
  -F tier=flash \
  -F pages=1-3,5 \
  -F lang=zh \
  -F parse_chart=true \
  -F keep_header_footer=false \
  -F include_page_markers=false
```

The call waits for the complete result and returns the parse result as the response body, with `Content-Type: application/json`. For an image, pass it as `file` and omit `pages`.

## `POST /v1/parse` as a stream

```bash
curl -N "$INFINITY_PARSER_SERVICE_URL/v1/parse" \
  -H "Authorization: Bearer $INFINITY_PARSER_API_KEY" \
  -F file=@report.pdf \
  -F tier=flash \
  -F pages=1-3,5
```

`-N` turns off curl's output buffering, so each event prints as it arrives. Leaving out `stream`, or sending `stream=true`, gives this response.

The response is `text/event-stream`. Each event is an `event:` line and a `data:` line holding single-line JSON, followed by a blank line. Every event carries `request_id`, which the `X-Request-ID` header repeats.

| Event | `data` |
| --- | --- |
| `progress` | sent first: `phase`, `model`, `total_pages`, `selected_pages`, `selected_page_count`, `completed_pages`, `progress`, `page_streaming` |
| `page_parsed` | `page`, that page's `nodes` and `md`, `total_pages`, `selected_page_count`, `completed_pages`, `progress`, `nodes_so_far` |
| `page_failed` | `page`, `error`, `error_type`, `status_code`, and the same progress fields |
| `heartbeat` | `elapsed_sec`; sent every 10 seconds without another event, and not a sign of progress |
| `done` | the full parse result; the stream ends |
| `failed` | `error`, `error_type`, `status_code`, `param`; the stream ends without `done` |

In `page_parsed`, every node has `parent_id` set to `null` and the event omits `logical_doctree`. Read the hierarchy from `done`, the `stream=false` response, or the chat endpoint’s parse result.

Page events arrive in completion order, so page 3 can precede page 1; treat `done` as the final, ordered result. A node that spans pages appears in each of those pages' events with the same `id`, and once in `done`. `nano` sends its page events only after the whole document is parsed and reports `page_streaming: false`.

Authentication, input and page-range errors are rejected before the stream starts, with a normal HTTP error. Once the stream starts the status is 200, so read the events: failed pages still end in `done`, while every selected page failing or the key being rejected upstream ends in `failed`. A stream that closes without `done` or `failed` is incomplete.

## `POST /v1/chat/completions`

The body is an OpenAI Chat Completions request with exactly one input content part: `file` for a PDF or `image_url` for an image. For a PDF, `file.file_data` must be a `data:application/pdf;base64,...` URL and `file.filename` must end in `.pdf`. For an image, `image_url.url` must be a `data:image/<format>;base64,...` URL. HTTP(S) URLs and `file_id` are not accepted. `stream` must be absent or false; use `/v1/parse` for per-page results. The call waits for the complete result.

```json
{
  "model": "infinity-parser-flash",
  "messages": [{
    "role": "user",
    "content": [{
      "type": "file",
      "file": {"filename": "document.pdf", "file_data": "data:application/pdf;base64,..."}
    }]
  }],
  "parser_options": {
    "pages": "1-3,5",
    "lang": "zh",
    "parse_chart": true,
    "keep_header_footer": false,
    "include_page_markers": false
  }
}
```

An image replaces the content part with:

```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
```

This builds that request for `report.pdf` in a file and sends it:

```bash
{
  printf '%s' '{"model":"infinity-parser-flash","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"report.pdf","file_data":"data:application/pdf;base64,'
  base64 < report.pdf | tr -d '\n'
  printf '%s' '"}}]}],"parser_options":{"pages":"1-3,5","lang":"zh","parse_chart":true,"keep_header_footer":false,"include_page_markers":false}}'
} > request.json

curl "$INFINITY_PARSER_SERVICE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $INFINITY_PARSER_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

The body goes in a file because the Base64 of a large PDF can exceed the shell's argument limit. `tr -d '\n'` removes the line breaks that `base64` inserts, which would make the JSON invalid.

`choices[0].message.content` holds the Markdown, and `parser_result` holds the rest of the parse result, `json` included; the client joins them back into the parse result. Setting `"response_format": {"type": "json_object"}` instead puts the whole parse result, `md` included, in `message.content` as a JSON string and omits `parser_result`. The completion's top-level `usage` writes unknown token counts as 0; read `parser_result.usage` for the counts that tell unknown from zero.

## Example (parsedoc 1.0)

This exchange was recorded with parsedoc 1.0, which omitted `logical_doctree` and returned `null` for every `parent_id`. The input is page 56 of a 324-page annual report filed with HKEX, the five-year financial summary:

```bash
curl -fL -o 2025062601065_c.pdf https://www1.hkexnews.hk/listedco/listconews/sehk/2025/0626/2025062601065_c.pdf

curl "$INFINITY_PARSER_SERVICE_URL/v1/parse" \
  -H "Authorization: Bearer $INFINITY_PARSER_API_KEY" \
  -F file=@2025062601065_c.pdf \
  -F stream=false \
  -F tier=flash \
  -F pages=56
```

The service answered HTTP 200 with `Content-Type: application/json` and this body, unchanged except for line breaks:

```json
{
  "schema_version": 3,
  "model": "infinity-parser-flash",
  "total_pages": 324,
  "parsed_pages": "56",
  "failed_pages": [],
  "json": {
    "type": "parsedoc",
    "version": "1.0",
    "meta": {
      "raw_type": "application/pdf",
      "raw_name": "2025062601065_c.pdf",
      "raw_md5": "af2ff981caded7033ebba24443b6cdeb",
      "raw_size": "4630060",
      "page_width": 595.2760009765625,
      "page_height": 841.8900146484375,
      "attrs": "{\"lang\": null}"
    },
    "data": {
      "nodes": [
        {
          "id": "p56-n1",
          "parent_id": null,
          "type": "apara",
          "category": "header",
          "content": "財務概要",
          "ct": "md",
          "refer": {
            "locs": [
              {
                "page": 56,
                "bbox": [0.767, 0.061, 0.905, 0.088]
              }
            ]
          },
          "size": null
        },
        {
          "id": "p56-n2",
          "parent_id": null,
          "type": "apara",
          "category": "table_caption",
          "content": "截至3月31日止年度",
          "ct": "md",
          "refer": {
            "locs": [
              {
                "page": 56,
                "bbox": [0.568, 0.138, 0.716, 0.151]
              }
            ]
          },
          "size": null
        },
        {
          "id": "p56-n3",
          "parent_id": null,
          "type": "atable",
          "category": "table",
          "content": "<table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024</td><td>2025</td></tr><tr><td></td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td></tr><tr><td></td><td colspan=\"5\">（百萬元）</td></tr><tr><td>收入</td><td>717,289</td><td>853,062</td><td>868,687</td><td>941,168</td><td>996,347</td></tr><tr><td>經營利潤</td><td>89,678</td><td>69,638</td><td>100,351</td><td>113,350</td><td>140,905</td></tr><tr><td>歸屬於普通股股東的淨利潤</td><td>150,308</td><td>61,959</td><td>72,509</td><td>79,741</td><td>129,470</td></tr><tr><td>淨利潤</td><td>143,284</td><td>47,079</td><td>65,573</td><td>71,332</td><td>125,976</td></tr></table>",
          "ct": "html",
          "refer": {
            "locs": [
              {
                "page": 56,
                "bbox": [0.093, 0.152, 0.904, 0.297]
              }
            ]
          },
          "size": null
        },
        {
          "id": "p56-n4",
          "parent_id": null,
          "type": "apara",
          "category": "table_caption",
          "content": "於3月31日",
          "ct": "md",
          "refer": {
            "locs": [
              {
                "page": 56,
                "bbox": [0.601, 0.313, 0.683, 0.328]
              }
            ]
          },
          "size": null
        },
        {
          "id": "p56-n5",
          "parent_id": null,
          "type": "atable",
          "category": "table",
          "content": "<table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024</td><td>2025</td></tr><tr><td></td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td></tr><tr><td></td><td colspan=\"5\">（百萬元）</td></tr><tr><td>流動資產</td><td>643,360</td><td>638,535</td><td>697,966</td><td>752,864</td><td>674,049</td></tr><tr><td>資產總額</td><td>1,690,218</td><td>1,695,553</td><td>1,753,044</td><td>1,764,829</td><td>1,804,227</td></tr><tr><td>流動負債</td><td>377,358</td><td>383,784</td><td>385,351</td><td>421,507</td><td>435,346</td></tr><tr><td>負債總額</td><td>606,584</td><td>613,360</td><td>630,123</td><td>652,230</td><td>714,121</td></tr><tr><td>股東權益</td><td>937,470</td><td>948,479</td><td>989,657</td><td>986,544</td><td>1,009,858</td></tr><tr><td>權益總額</td><td>1,074,961</td><td>1,072,538</td><td>1,113,063</td><td>1,101,871</td><td>1,078,393</td></tr></table>",
          "ct": "html",
          "refer": {
            "locs": [
              {
                "page": 56,
                "bbox": [0.093, 0.328, 0.904, 0.513]
              }
            ]
          },
          "size": null
        },
        {
          "id": "p56-n6",
          "parent_id": null,
          "type": "apara",
          "category": "footer",
          "content": "2025財務年度報告",
          "ct": "md",
          "refer": {
            "locs": [
              {
                "page": 56,
                "bbox": [0.777, 0.955, 0.904, 0.969]
              }
            ]
          },
          "size": null
        },
        {
          "id": "p56-n7",
          "parent_id": null,
          "type": "apara",
          "category": "footer",
          "content": "55",
          "ct": "md",
          "refer": {
            "locs": [
              {
                "page": 56,
                "bbox": [0.928, 0.956, 0.949, 0.968]
              }
            ]
          },
          "size": null
        }
      ]
    }
  },
  "md": "截至3月31日止年度\n\n<table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024</td><td>2025</td></tr><tr><td></td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td></tr><tr><td></td><td colspan=\"5\">（百萬元）</td></tr><tr><td>收入</td><td>717,289</td><td>853,062</td><td>868,687</td><td>941,168</td><td>996,347</td></tr><tr><td>經營利潤</td><td>89,678</td><td>69,638</td><td>100,351</td><td>113,350</td><td>140,905</td></tr><tr><td>歸屬於普通股股東的淨利潤</td><td>150,308</td><td>61,959</td><td>72,509</td><td>79,741</td><td>129,470</td></tr><tr><td>淨利潤</td><td>143,284</td><td>47,079</td><td>65,573</td><td>71,332</td><td>125,976</td></tr></table>\n\n於3月31日\n\n<table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024</td><td>2025</td></tr><tr><td></td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td><td>人民幣</td></tr><tr><td></td><td colspan=\"5\">（百萬元）</td></tr><tr><td>流動資產</td><td>643,360</td><td>638,535</td><td>697,966</td><td>752,864</td><td>674,049</td></tr><tr><td>資產總額</td><td>1,690,218</td><td>1,695,553</td><td>1,753,044</td><td>1,764,829</td><td>1,804,227</td></tr><tr><td>流動負債</td><td>377,358</td><td>383,784</td><td>385,351</td><td>421,507</td><td>435,346</td></tr><tr><td>負債總額</td><td>606,584</td><td>613,360</td><td>630,123</td><td>652,230</td><td>714,121</td></tr><tr><td>股東權益</td><td>937,470</td><td>948,479</td><td>989,657</td><td>986,544</td><td>1,009,858</td></tr><tr><td>權益總額</td><td>1,074,961</td><td>1,072,538</td><td>1,113,063</td><td>1,101,871</td><td>1,078,393</td></tr></table>",
  "request_id": "4ab4c746176f4db696e0e5ed4a399d80",
  "usage": {
    "pages_processed": 1,
    "prompt_tokens": 8814,
    "completion_tokens": 1229,
    "total_tokens": 10043,
    "usage_complete": true,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "cache_usage_complete": true
  },
  "warnings": []
}
```

What a correct call looks like here:

- `schema_version` is 3. `total_pages` is the whole document, 324, while `parsed_pages` is only the selected page, as the string `"56"`, and `failed_pages` is empty.
- `json.meta.raw_name` is the filename the upload carried, and `raw_md5` and `raw_size` match the whole PDF, not the selected page.
- `json.data.nodes` include the page's header (`財務概要`) and footers (`2025財務年度報告`, `55`), but `md` leaves them out, because `keep_header_footer` defaults to `false`. The tables are `atable` nodes holding HTML, with `ct` `html`.
- `request_id` matches the `X-Request-ID` response header and differs on every call. `cached_tokens` depends on what the service has cached, and the parsed content can change slightly between service versions.

The same request without `stream=false` returned HTTP 200 with `Content-Type: text/event-stream; charset=utf-8` and three events. The first was:

```text
event: progress
data: {"phase": "parse", "model": "infinity-parser-flash", "total_pages": 324, "selected_pages": [56], "selected_page_count": 1, "completed_pages": 0, "progress": 0.0, "page_streaming": true, "request_id": "1805cc7aaeff40268d6d14b93ff956d3"}
```

Next came `page_parsed` for page 56 with the same 7 nodes, `p56-n1` to `p56-n7`, and its `md`, then `done` with the body above, apart from `request_id` and `cached_tokens`.

The same page sent to `/v1/chat/completions` as the request in that section, with `"parser_options": {"pages": "56"}`, returned a completion whose `choices[0].message.content` equals the `md` above, and whose `parser_result` holds every other field above, with the same `json`.

## Errors

| Status | Meaning |
| --- | --- |
| 400 | Invalid JSON, an unsupported or invalid file, a multi-frame image, an unsupported tier or model, invalid options, a page range outside the document, or more selected pages than the per-request limit |
| 401/403 | Missing or rejected API key. A 403 `invalid inference serving id: record not found` comes from the platform gateway and means no service is deployed at that URL |
| 413 | The file exceeds 50 MiB. The chat endpoint refuses a body larger than the Base64 of 50 MiB plus 1 MiB as soon as it reads `Content-Length`, and can close the connection before the whole body is sent, so the caller sees a reset instead of the 413 |
| 422 | A multipart request without `file` or with a malformed form value |
| 429/503 | Too many requests from this caller, or the service is busy; retry later |
| 502 | Upstream failure, including every selected page failing |

Errors use `{"error":{"message":"...","type":"...","param":"pages","code":null}}`; `param` names the offending field when known and is otherwise `null`. A 422 uses FastAPI's `{"detail":[...]}` shape instead. A 429, or a 503 because the service is busy, carries `Retry-After: 5`; inside a stream, the `failed` or `page_failed` event carries `retry_after: 5`.
