import re
from typing import Pattern


class LatexError(Exception):
    pass


class LatexValidationError(LatexError):
    pass


class BracketMismatchError(LatexValidationError):
    pass


class EnvironmentMismatchError(LatexValidationError):
    pass


class InvalidCharacterError(LatexValidationError):
    pass


class LatexSimplificationError(LatexError):
    pass


class LatexValidator:
    _invalid_unicode_re: Pattern[str] = re.compile(r"[\u0000-\u001F\u007F]")
    _env_token_re: Pattern[str] = re.compile(r"\\(begin|end)\{([^\}]+)\}")
    _illegal_backslash_re: Pattern[str] = re.compile(r"(\\[^a-zA-Z])")
    _allowed_non_letter_prefixes = {
        "\\\\",
        "\\[",
        "\\]",
        "\\(",
        "\\)",
        "\\%",
        "\\&",
        "\\$",
        "\\#",
        "\\,",
        "\\;",
        "\\:",
        "\\!",
        "\\ ",
        "\\quad",
        "\\qquad",
    }

    def __call__(self, latex: str) -> bool:
        return self.is_valid(latex)

    def is_valid(self, latex: str) -> bool:
        if not latex or not isinstance(latex, str):
            raise LatexValidationError("Input is empty or not a string.")

        for i, line in enumerate(latex.splitlines(), start=1):
            if self._invalid_unicode_re.search(line):
                snippet = repr(line.strip())[:60]
                raise InvalidCharacterError(
                    f"Line {i} contains invalid Unicode control characters: {snippet}"
                )

        if self._has_illegal_backslashes(latex):
            raise InvalidCharacterError("Contains illegal backslash usage.")

        if not self._are_brackets_balanced(latex, "{", "}"):
            raise BracketMismatchError("Mismatched {} brackets.")
        if not self._are_brackets_balanced(latex, "[", "]"):
            raise BracketMismatchError("Mismatched [] brackets.")
        if not self._are_brackets_balanced(latex, "(", ")"):
            raise BracketMismatchError("Mismatched () brackets.")
        if not self._are_environments_balanced(latex):
            raise EnvironmentMismatchError("Environment \\begin/\\end mismatch.")
        return True

    def _are_brackets_balanced(self, s: str, open_b: str, close_b: str) -> bool:
        stack = []
        for c in s:
            if c == open_b:
                stack.append(c)
            elif c == close_b:
                if not stack:
                    return False
                stack.pop()
        return not stack

    def _are_environments_balanced(self, s: str) -> bool:
        tokens = self._env_token_re.findall(s)
        stack = []
        for kind, name in tokens:
            if kind == "begin":
                stack.append(name)
            elif kind == "end":
                if not stack or stack[-1] != name:
                    return False
                stack.pop()
        return not stack

    def _has_illegal_backslashes(self, s: str) -> bool:
        for match in self._illegal_backslash_re.findall(s):
            if match not in self._allowed_non_letter_prefixes:
                return True
        return False


class LatexSimplifier:
    _whitespace_re: Pattern[str] = re.compile(r"\s+")
    _operator_spacing_re: Pattern[str] = re.compile(r"\s*([=+\-*/<>])\s*")
    _inline_wrap_re: Pattern[str] = re.compile(r"^\$(.*?)\$$", re.DOTALL)
    _display_wrap_re: Pattern[str] = re.compile(r"^\$\$(.*?)\$\$$", re.DOTALL)
    _bracket_wrap_re: Pattern[str] = re.compile(r"^\\[\[\(](.*?)\\[\]\)]$", re.DOTALL)
    _text_expr_re: Pattern[str] = re.compile(r"\\text\{.*?\}")
    _operator_expr_re: Pattern[str] = re.compile(r"\\operatorname\{.*?\}")
    _structure_spacing_re = re.compile(r"\s*(\\(?:begin|end)\{[^\}]+\})\s*")
    # _old_style_font_re: Pattern[str] = re.compile(r"(\\(?:bf|it|rm|tt|sf|sl|sc))\s+")
    _backslash_spacing_re = re.compile(r"(\\)\s")
    _cmd_spacing_re = re.compile(r"(\\[a-zA-Z]+)\s+(?=[a-zA-Z])")
    _all_space_re = re.compile(r"\s+")

    @staticmethod
    def _protect_space(m) -> str:
        return m.group(0).replace(" ", "␣")

    @staticmethod
    def _protect_oldstylefontspace(m) -> str:
        return m.group(1) + "␣"

    def remove_wrappers(self, latex: str) -> str:
        latex = latex.strip()
        for pattern in [
            self._display_wrap_re,
            self._inline_wrap_re,
            self._bracket_wrap_re,
        ]:
            match = pattern.match(latex)
            if match:
                return match.group(1).strip()
        return latex

    def compress_whitespace(self, latex: str) -> str:

        latex = self._text_expr_re.sub(LatexSimplifier._protect_space, latex)
        latex = self._operator_expr_re.sub(LatexSimplifier._protect_space, latex)

        latex = self._backslash_spacing_re.sub(r"\1␣", latex)

        latex = self._cmd_spacing_re.sub(r"\1␣", latex)

        latex = self._all_space_re.sub("", latex)

        latex = latex.replace("␣", " ")
        return latex


class LatexNormalizer:
    def __init__(
        self,
        *,
        strip_wrappers: bool = True,
        flatten_multiline_to_single_line: bool = True,
        simplify_whitespace: bool = True,
        validate: bool = True,
    ) -> None:
        self.strip_wrappers = strip_wrappers
        self.flatten_multiline_to_single_line = flatten_multiline_to_single_line
        self.simplify_whitespace = simplify_whitespace
        self.validate = validate

        self._validator = LatexValidator()
        self._simplifier = LatexSimplifier()

    def __call__(self, latex: str) -> str:
        if not isinstance(latex, str):
            raise LatexValidationError("Input is not a string.")

        if self.strip_wrappers:
            latex = self._simplifier.remove_wrappers(latex)

        if self.flatten_multiline_to_single_line:
            lines = [line.strip() for line in latex.splitlines() if line.strip()]
            latex = " ".join(lines)

        if self.simplify_whitespace:
            latex = self._simplifier.compress_whitespace(latex)

        if self.validate:
            self._validator(latex)
        return latex
