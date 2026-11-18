"""HTTP helpers for parse.py: the service URL check, the request body read from
disk as it is sent, server-sent events, error details, and refusing redirects."""

import base64
import http.client
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request

# The body is sent in chunks of this size. Each is a separate socket write with
# its own timeout, so a slow uplink can't run out the timeout during a long
# upload. A multiple of 3, so the Base64 of the chunks joins without padding.
UPLOAD_CHUNK_BYTES = 3 * 256 * 1024
ERROR_DETAIL_CHARS = 300


def loopback(host):
    """Whether `host`, as urlsplit() gives it, is this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def url_problem(url):
    """Why `url` can't be the service's URL, or None if it can."""
    try:
        parts = urllib.parse.urlsplit(url)
        # These raise ValueError for a port that isn't a number in range, or a
        # host name with an empty or overlong label, which would fail the request.
        parts.port
        (parts.hostname or "").encode("idna")
    except ValueError:
        return "is not a valid URL"
    if parts.scheme.lower() not in ("http", "https"):
        return "must be an http(s) URL"
    if not parts.hostname or any(char.isspace() or not char.isprintable() for char in url):
        return "is not a valid URL"
    # The endpoint's path is added to the end of the URL, so it would land in these.
    if "?" in url or "#" in url:
        return "can't have a query (?) or fragment (#)"
    # The key goes with the first request, so refusing a redirect to https
    # afterwards would be too late.
    if parts.scheme.lower() == "http" and not loopback(parts.hostname):
        return "must be an https URL, so that the API key isn't sent unencrypted; http is only for localhost"
    return None


class Body:
    """`head`, the first `size` bytes of the file at `path`, then `tail`, read
    from disk while the request is sent. http.client writes each chunk with its
    own socket call, so the timeout bounds a chunk rather than the whole upload,
    and the document is never held in memory. With `b64` the file is sent as
    Base64. Each iteration sends the body from the start."""

    def __init__(self, head, path, size, tail, b64=False):
        self.head, self.path, self.size, self.tail, self.b64 = head, path, size, tail, b64
        self.length = len(head) + ((size + 2) // 3 * 4 if b64 else size) + len(tail)
        # http.client sends the body once it has connected and sent the headers,
        # so from then on a failure is the upload's, not the connection's.
        self.started = False

    def __iter__(self):
        self.started = True
        yield self.head
        with self.path.open("rb") as file:
            left = self.size
            while left:
                chunk = file.read(min(UPLOAD_CHUNK_BYTES, left))
                if not chunk:
                    raise OSError(f"{self.path} got shorter while it was being sent")
                left -= len(chunk)
                yield base64.b64encode(chunk) if self.b64 else chunk
        yield self.tail


def sse_events(lines):
    """Yield (event, data) per complete SSE frame; a frame cut off by EOF is dropped."""
    event, data = "message", []
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)


def retry_hint(seconds):
    return f"; retry after {seconds} s" if seconds else ""


def error_detail(exc):
    """The message in an HTTP error's body, or the body itself if it isn't the
    service's JSON; on one line, cut to ERROR_DETAIL_CHARS."""
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except (OSError, http.client.HTTPException):
        # The body stalled or was cut off; the status line is all there is.
        return exc.reason
    try:
        payload = json.loads(body)
        error = payload.get("error")
        # The service's {"error": {"message": ...}}, or a gateway's {"error": "..."}.
        message = error.get("message") if isinstance(error, dict) else error if isinstance(error, str) else None
        detail = message or payload.get("detail") or body
    except (ValueError, AttributeError):
        # The platform gateway answers in plain text, e.g. "Request denied by Inspire
        # Auth check", and a proxy can answer with a whole HTML page.
        detail = body or exc.reason
    detail = " ".join(str(detail).split())
    return detail if len(detail) <= ERROR_DETAIL_CHARS else detail[:ERROR_DETAIL_CHARS] + "..."


class NoRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect. urllib's default handler re-sends the Authorization
    header to wherever a redirect points, even another host or an https -> http
    downgrade, and it would resend the POST as a GET without the document."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError(f"{req.full_url} answered with a {code} redirect to {newurl}; not followed")


OPENER = urllib.request.build_opener(NoRedirects)
