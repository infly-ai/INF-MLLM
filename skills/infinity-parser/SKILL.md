---
name: infinity-parser
description: Parse a PDF or image with the hosted Infinity Parser service and save structured JSON plus Markdown. Use for document extraction, tables, charts, and page-level parsing.
---

# Infinity Parser

Use the bundled standard-library client, which needs Python 3.10 or newer. It reads
`INFINITY_PARSER_SERVICE_URL` and `INFINITY_PARSER_API_KEY` from the environment or
`~/.config/inftech/.env`; `INF_API_KEY` is accepted as a fallback key. The default
service URL is `https://parser-gw.openapi-inspire.inf.tech`.

To get an API key, sign in at <https://agent-infinity-parser.openapi-inspire.inf.tech>
and apply for one there, then save it in `~/.config/inftech/.env` as
`INFINITY_PARSER_API_KEY=<key>`. If no key is set, point the user to that page.

The service has two parse endpoints, each authenticated with
`Authorization: Bearer <key>`. The client's `--endpoint` option picks one:

- `parse` (the default): `POST /v1/parse`, a `multipart/form-data` upload
  answered with server-sent events per page, ending in a `done` or `failed` event.
- `chat`: `POST /v1/chat/completions`, an OpenAI Chat Completions request carrying
  one Base64 PDF or image.

Both save the same outputs and use the same exit codes. Keep the default unless
the user asks for the chat endpoint: the stream prints each page to stderr as it
finishes, and its heartbeats let the client tell a long parse from a dead connection.
The chat endpoint says nothing until the whole document is parsed, and the client
waits up to 30 minutes for it. See `references/api-reference.md` for each endpoint's
request and response format.

A parse can outlast a foreground command, which Claude Code kills after 2 minutes by
default and 10 at most. The service then sees the connection close and cancels the
parse, and nothing is saved. Unless the input is an image or a PDF of a few pages,
run the client as a background command (in Claude Code, the Bash tool's
`run_in_background`) and wait for it to exit.

From this skill directory:

```bash
python3 scripts/parse.py /path/to/report.pdf
python3 scripts/parse.py /path/to/born-digital.pdf --tier nano
python3 scripts/parse.py /path/to/report.pdf --output /tmp/parsed --pages 1-3
python3 scripts/parse.py /path/to/report.pdf --lang en --overwrite
python3 scripts/parse.py /path/to/scan.png
python3 scripts/parse.py /path/to/report.pdf --endpoint chat
```

The command writes `<stem>.json` and `<stem>.md` next to the input by default.
`<stem>.json` is the whole parse result: `parsed_pages` (a page-range string such as
`"1-3,5"`), `failed_pages`, `usage`, `warnings`, the Markdown as `md`, and the
document structure as `json`, whose `json.data.nodes` lists each paragraph, table
(HTML) or figure in reading order with its page and normalized bounding box in
`refer.locs`. To walk the document by section, follow `json.logical_doctree` in the
saved result. Each entry below the root identifies a node in `json.data.nodes` by `id`, and
each node's `parent_id` gives its parent in the hierarchy.

If either file already exists, it refuses before sending anything unless `--overwrite`
is provided. It also refuses if the output directory can't be written, or if an
output would replace the input (an input named `scan.md`, say); pass `--output` with
another directory. Without `--overwrite`, it also never replaces outputs that appear
while it parses, such as another run's outputs for an input of the same name, so give
parallel runs on same-named inputs different `--output` directories. It accepts PDFs
and single-frame PNG, JPEG, WEBP, BMP, TIFF or GIF images of up to 50 MiB, and refuses
a larger file before sending it; convert other documents to PDF first. An image is
one page, so omit `--pages` or pass `1`. With the stream, it prints lines such as
`Page 3 parsed (2/5)` to stderr; `nano` reports its pages only once the whole
document is parsed. It reports the server's error detail for failed requests, with a
retry hint when the service is busy. If it says the upload stopped with a broken pipe
or a reset, the service most likely refused the request before reading the whole
file, usually because the key is missing or invalid; check the key before retrying.
It refuses any redirect, so the key never leaves the service; if it reports one,
check `INFINITY_PARSER_SERVICE_URL` rather than pointing the client at the new address.
Supported options include `--tier` (`nano`, `flash`, `pro`), `--pages`, and `--lang`
(`zh` or `en`, the language of figure descriptions; default `zh`). Three options take
an explicit `true` or `false` value, as in `--parse-chart false`:
`--keep-header-footer` (default `false`), `--parse-chart` (default `true`), and
`--include-page-markers` (default `false`; a `%%% Page N` line where each page starts).

Choose the tier by input. Prefer `nano` for born-digital PDFs that have a text
layer. Use `flash` (the default) or `pro` for images and scanned PDFs: `nano` can
return empty Markdown and no nodes for them while still exiting 0. If a `nano`
result is empty, rerun with `--tier flash --overwrite`.

Exit codes:

- 0: every page parsed.
- 1: the request failed, including a stream that ended in `failed` or was cut off;
  nothing was written.
- 2: usage error, including an unsupported or oversized input, existing outputs, or
  outputs that can't be saved; nothing was sent.
- 3: the outputs were saved, but some pages failed. Report the `failed_pages` it prints.
- 4: the document was parsed, but the outputs couldn't be saved in the output
  directory, for example because the disk filled up or another run saved them first.
  Nothing in the output directory was changed, unless stderr says an output can't be
  put back as it was; it then names the hidden file that holds the previous output.
  stderr names the temporary directory that holds the outputs instead; tell the user
  where they are rather than rerunning, which would parse the document again.

The client refuses a service URL that isn't `https`, except on localhost, so the key
is never sent unencrypted. Don't disable TLS verification. Keep the API key in the
user configuration file, not in the skill directory.
