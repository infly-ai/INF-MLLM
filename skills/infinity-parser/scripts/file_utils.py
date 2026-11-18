"""Local-file helpers for parse.py: the user's .env file, the input's media type,
and saving the outputs without leaving a partial or mismatched pair."""

import contextlib
import os
import stat
import tempfile
from pathlib import Path

IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"II+\x00", "image/tiff"),  # BigTIFF
    (b"MM\x00+", "image/tiff"),
)
# A BMP's 14-byte file header is followed by a DIB header that starts with its
# own size; these are the sizes image decoders accept.
BMP_HEADER_SIZES = (12, 40, 52, 56, 64, 108, 124)


# The compliance client's .env rules, so a line means the same to every skill.
def env_file_value(key):
    path = Path.home() / ".config" / "inftech" / ".env"
    try:
        # UTF-8 whatever the locale, which on Windows is an ANSI code page, and
        # without the byte-order mark that Notepad or PowerShell can put first.
        for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    except OSError:
        pass
    return None


def inspect_input(path):
    """Return the file's size and media type; raise OSError if it can't be read."""
    with path.open("rb") as file:
        return os.fstat(file.fileno()).st_size, media_type(file.read(1024))


def media_type(content):
    """The media type of a file that starts with `content`, or None if unsupported."""
    # The service's own rule: it refuses a header further in, which the PDF spec allows.
    if content[:1024].lstrip().startswith(b"%PDF-"):
        return "application/pdf"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    # "BM" alone also starts plain text, such as "BMW annual report".
    if content[:2] == b"BM" and int.from_bytes(content[14:18], "little") in BMP_HEADER_SIZES:
        return "image/bmp"
    for signature, mime in IMAGE_SIGNATURES:
        if content.startswith(signature):
            return mime
    return None


def check_writable(output, paths):
    """Raise OSError now, before the parse is paid for, if `paths` can't be saved in `output`."""
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output):
        pass
    for path in paths:
        # Only with --overwrite. A new file is renamed over the output, which takes
        # the directory checked above rather than a writable output, and would
        # replace a symlink itself rather than the file it points to.
        if path.is_symlink() or path.exists() and not path.is_file():
            raise OSError(f"{path} is a symlink or not a file; --overwrite only replaces files")


def create(path, data, mode=0o666):
    """Write `data` to `path`, which must not exist yet, and return `path`; the
    file is created with `mode`, less the umask. A failed write removes the
    file rather than leave it cut short."""
    file = open(path, "xb", opener=lambda name, flags: os.open(name, flags, mode))
    try:
        with file:
            file.write(data)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return path


def hidden(path, suffix):
    """A new name for a hidden file next to `path`."""
    return path.with_name(f".{path.name}.{os.urandom(8).hex()}{suffix}")


def stage(path, data):
    """Write `data` to a new hidden file next to `path`, with the permissions of
    the output it is to replace, and return the file's path."""
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        # No output yet, so the file gets the permissions any new file gets.
        return create(hidden(path, ".tmp"), data)
    # Created with them, so the result is never more readable than the output,
    # even while it is written; the chmod adds back what the umask took away.
    temp = create(hidden(path, ".tmp"), data, mode)
    try:
        os.chmod(temp, mode)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise
    return temp


def set_aside(path):
    """Move the output file at `path` to a new hidden name next to it, and
    return that name; or None if there is no output file."""
    if not path.is_file():
        return None
    aside = hidden(path, ".old")
    os.replace(path, aside)
    return aside


def replace_outputs(staged):
    """Move each (temp, path) to `path`. If one can't be moved, put back the
    outputs already replaced, so that they never disagree, and raise; the error
    names any output that can't be put back."""
    # How to undo each step so far: move the previous output back to `path`,
    # or with None remove this run's `path`, which had no previous output.
    undo = []
    try:
        for temp, path in staged:
            previous = set_aside(path)
            if previous:
                undo.append((path, previous))
            os.replace(temp, path)
            if not previous:
                undo.append((path, None))
    except BaseException as exc:
        stuck = []
        for path, previous in reversed(undo):
            try:
                if previous:
                    os.replace(previous, path)
                else:
                    os.unlink(path)
            except OSError as undo_exc:
                kept = f"; the previous one is {previous}" if previous else ""
                stuck.append(f"{path} can't be put back as it was: {undo_exc}{kept}")
        if stuck and isinstance(exc, OSError):
            raise OSError("; ".join([str(exc), *stuck])) from exc
        raise
    for _path, previous in undo:
        if previous:
            with contextlib.suppress(OSError):
                os.unlink(previous)


def save(outputs, overwrite):
    """Write each (path, data), or raise OSError; a failed write changes no
    output, unless the error says it can't put one back. Without `overwrite`,
    an output that exists by now, say because another run saved it during the
    parse, raises FileExistsError and is left alone."""
    made = []  # the new outputs, or with `overwrite` the files to replace the outputs with
    try:
        for path, data in outputs:
            made.append(stage(path, data) if overwrite else create(path, data))
        if overwrite:
            # Every output is written in full before any is replaced.
            replace_outputs([(temp, path) for temp, (path, _data) in zip(made, outputs)])
    except BaseException:
        for path in made:
            with contextlib.suppress(OSError):
                os.unlink(path)
        raise


def save_elsewhere(outputs):
    """Save the outputs in a new temporary directory and return it, so that a
    parse that was paid for isn't lost when the output directory fails."""
    directory = Path(tempfile.mkdtemp(prefix="infinity-parser-"))
    save([(directory / path.name, data) for path, data in outputs], overwrite=False)
    return directory
