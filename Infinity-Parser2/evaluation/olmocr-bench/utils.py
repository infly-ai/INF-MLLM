"""Post-processing utilities for parsed markdown.

Bundles three helpers used by the olmocr-bench inference pipeline:
- ``apply_synonym_map``            : normalize common symbol variants
- ``convert_latex_in_markdown``    : convert LaTeX formulas to Unicode
- ``latex_formula_normalization``  : normalize/merge adjacent LaTeX formulas
"""

import re
from typing import Callable


# ===========================================================================
# Synonym mapping
# ===========================================================================

# Synonym mapping table: src -> target
SYNONYM_MAP: dict[str, str] = {
    "☐": "□",  # ☐ -> □
    "–": "-",        # – (en-dash) -> -
    r"\cdot": ".",        # \cdot -> .
    "◇": "◊",   # ◇ -> ◊
}


def apply_synonym_map(text: str) -> str:
    """Replace all synonyms in the text with their target forms.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text after replacement.
    """
    for src, target in SYNONYM_MAP.items():
        text = text.replace(src, target)
    return text


def get_synonym_mapper() -> Callable[[str], str]:
    """Return the synonym mapping function, equivalent to calling ``apply_synonym_map`` directly.

    Returns
    -------
    Callable[[str], str]
        A function usable in chained calls such as ``.map()``.
    """
    return apply_synonym_map


# ===========================================================================
# LaTeX to Unicode conversion
# ===========================================================================

# LaTeX to Unicode mapping table
LATEX_TO_UNICODE = {
    # Greek letters (lowercase)
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
    r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
    r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
    r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
    r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',

    # Greek letters (uppercase)
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
    r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',

    # Math operators
    r'\pm': '±', r'\mp': '∓', r'\times': '×', r'\div': '÷',
    r'\cdot': '.', r'\ast': '∗', r'\star': '⋆', r'\circ': '∘',
    r'\bullet': '•', r'\oplus': '⊕', r'\ominus': '⊖', r'\otimes': '⊗',
    r'\odot': '⊙', r'\dagger': '†', r'\ddagger': '‡',

    # Relational operators
    r'\leq': '≤', r'\le': '≤', r'\geq': '≥', r'\ge': '≥',
    r'\neq': '≠', r'\ne': '≠', r'\approx': '≈', r'\equiv': '≡',
    r'\cong': '≅', r'\sim': '∼', r'\simeq': '≃', r'\propto': '∝',
    r'\ll': '≪', r'\gg': '≫', r'\prec': '≺', r'\succ': '≻',
    r'\perp': '⊥', r'\parallel': '∥',

    # Set operators
    r'\subset': '⊂', r'\supset': '⊃', r'\subseteq': '⊆', r'\supseteq': '⊇',
    r'\in': '∈', r'\notin': '∉', r'\cup': '∪', r'\cap': '∩',
    r'\emptyset': '∅', r'\varnothing': '∅', r'\land': '∧', r'\lor': '∨',

    # Arrow symbols
    r'\rightarrow': '→', r'\to': '→', r'\leftarrow': '←',
    r'\Rightarrow': '⇒', r'\Leftarrow': '⇐', r'\leftrightarrow': '↔',
    r'\mapsto': '↦', r'\nearrow': '↗', r'\searrow': '↘',
    r'\swarrow': '↙', r'\nwarrow': '↖',

    # Other math symbols
    r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇',
    r'\forall': '∀', r'\exists': '∃', r'\neg': '¬',
    r'\angle': '∠', r'\measuredangle': '∡', r'\triangle': '△',
    r'\blacksquare': '■', r'\square': '□', r'\diamond': '◇',
    r'\clubsuit': '♣', r'\diamondsuit': '♢', r'\heartsuit': '♡', r'\spadesuit': '♠',

    # Logic symbols
    r'\therefore': '∴', r'\because': '∵', r'\qed': '∎',

    # Number modifiers
    r'\prime': '′', r'\dprime': '″',

    # Integral symbols
    r'\oint': '∮', r'\iiint': '∭', r'\iint': '∬',
}


def latex_to_unicode(text):
    """Convert LaTeX formulas to Unicode symbols."""
    if text is None:
        return text

    result = text

    # Step 1: handle parameterized commands like Blackboard Bold \mathbb{R} -> ℝ
    blackboard_bold_map = {
        'A': '𝔸', 'B': '𝔹', 'C': 'ℂ', 'D': '𝔻', 'E': '𝔼', 'F': '𝔽',
        'G': '𝔾', 'H': 'ℍ', 'I': '𝕀', 'J': '𝕁', 'K': '𝕂', 'L': '𝕃',
        'M': '𝕄', 'N': 'ℕ', 'O': '𝕆', 'P': 'ℙ', 'Q': 'ℚ', 'R': 'ℝ',
        'S': '𝕊', 'T': '𝕋', 'U': '𝕌', 'V': '𝕍', 'W': '𝕎', 'X': '𝕏',
        'Y': '𝕐', 'Z': 'ℤ',
        'a': '𝕒', 'b': '𝕓', 'c': '𝕔', 'd': '𝕕', 'e': '𝕖', 'f': '𝕗',
        'g': '𝕘', 'h': '𝕙', 'i': '𝕚', 'j': '𝕛', 'k': '𝕜', 'l': '𝕝',
        'm': '𝕞', 'n': '𝕟', 'o': '𝕠', 'p': '𝕡', 'q': '𝕢', 'r': '𝕣',
        's': '𝕤', 't': '𝕥', 'u': '𝕦', 'v': '𝕧', 'w': '𝕨', 'x': '𝕩',
        'y': '𝕪', 'z': '𝕫',
    }
    for char, unicode_char in blackboard_bold_map.items():
        result = re.sub(rf'\\mathbb\{{{char}\}}(?![a-zA-Z])', unicode_char, result)

    # Step 2: handle other LaTeX commands
    # Sort by descending length so that longer commands match first
    sorted_latex_commands = sorted(LATEX_TO_UNICODE.items(), key=lambda x: len(x[0]) if x[0] else 0, reverse=True)

    for latex, unicode_char in sorted_latex_commands:
        if latex and latex.startswith('\\'):
            # Match the full LaTeX command, followed by a non-letter char or end of string
            pattern = re.escape(latex) + r'(?![a-zA-Z])'
            result = re.sub(pattern, unicode_char, result)
        elif latex:
            result = result.replace(latex, unicode_char)

    # Handle fractions \frac{a}{b} -> a/b
    result = re.sub(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', r'\1/\2', result)

    # Handle \sqrt[n]{x} -> ⁿ√x (nth root)
    result = re.sub(r'\\sqrt\s*\[(\d+)\]\s*\{([^}]+)\}', r'⁽\1⁾√\2', result)
    result = re.sub(r'\\sqrt\s*\{([^}]+)\}', r'√\1', result)

    # Handle \text{...} -> keep the content inside the braces, stripping surrounding whitespace
    result = re.sub(r'\\text\s*\{([^}]+)\}', lambda m: m.group(1).strip(), result)
    # Remove extra whitespace introduced by the \text conversion
    result = re.sub(r'\{\s*', '{', result)
    result = re.sub(r'\s*}', '}', result)

    # Special case: \text{content1}^{content2} form; strip whitespace around content1 then convert to superscript
    # Must run before the generic superscript handling so that \text is processed first
    def handle_text_superscript(match):
        text_content = match.group(1).strip()
        superscript_content = match.group(2)
        return text_content + convert_superscript(re.search(r'^\{([^}]+)\}$', superscript_content) or
                                                   re.search(r'^\{\{([^}]+)\}\}\}$', superscript_content) or
                                                   match)

    result = re.sub(r'\\text\s*\{([^}]+)\}\s*\^\{([^}]+)\}', handle_text_superscript, result)

    # Handle super/subscripts ^{n} -> ⁿ, _{n} -> ₙ, ^{xyz} -> ᵞᶻ
    # Special case: ^{th} ->  th (without the space it would be turned into a superscript)
    result = re.sub(r'\^\{th\}', ' th', result)
    result = re.sub(r'\^\{\{th\}\}', ' th', result)
    def convert_superscript(match):
        content = match.group(1)
        superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                       '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                       '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
                       'n': 'ⁿ', 'i': 'ⁱ', 'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ',
                       'd': 'ᵈ', 'e': 'ᵉ', 'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ',
                       'j': 'ʲ', 'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'o': 'ᵒ',
                       'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
                       'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
                       'th': 'ᵗʰ', 'st': 'ˢᵗ', 'nd': 'ⁿᵈ', 'rd': 'ʳᵈ'}
        # Try converting the whole content at once
        if content in superscripts:
            return superscripts[content]
        # Otherwise convert character by character
        return ''.join(superscripts.get(c, c) for c in content)

    def convert_subscript(match):
        content = match.group(1)
        subscripts = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                      '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                      'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
                      'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
                      'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
                      'v': 'ᵥ', 'x': 'ₓ'}
        # Only convert convertible characters, keep others (like -, =, (), etc.)
        return ''.join(subscripts.get(c, c) for c in content)

    result = re.sub(r'\^{\{([^}]+)\}\}', convert_superscript, result)
    result = re.sub(r'\^\{([^}]+)\}', convert_superscript, result)  # handle ^{content}
    result = re.sub(r'\^([^{])', convert_superscript, result)
    result = re.sub(r'_\{([^}]+)\}', convert_subscript, result)
    result = re.sub(r'_([a-zA-Z0-9])', convert_subscript, result)

    # Handle \left( and \right), etc.
    result = re.sub(r'\\left\s*\(', '(', result)
    result = re.sub(r'\\right\s*\)', ')', result)
    result = re.sub(r'\\left\s*\[', '[', result)
    result = re.sub(r'\\right\s*\]', ']', result)
    result = re.sub(r'\\left\s*\{', '{', result)
    result = re.sub(r'\\right\s*\}', '}', result)
    result = re.sub(r'\\left\s*\.', '', result)
    result = re.sub(r'\\right\s*\.', '', result)

    # Remove empty {}
    result = re.sub(r'\{\s*\}', '', result)

    # Handle \sum, \prod, \coprod
    result = re.sub(r'\\sum', '∑', result)
    result = re.sub(r'\\prod', '∏', result)
    result = re.sub(r'\\coprod', '∐', result)

    # Handle \int (various integral forms)
    result = re.sub(r'\\int', '∫', result)
    result = re.sub(r'\\iint', '∬', result)
    result = re.sub(r'\\iiint', '∭', result)
    result = re.sub(r'\\oint', '∮', result)

    # Handle limits \lim
    result = re.sub(r'\\lim', 'lim', result)

    # Handle common math functions
    math_functions = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                      'log', 'ln', 'exp', 'max', 'min', 'det', 'dim', 'ker', 'deg']
    for func in math_functions:
        result = re.sub(rf'\\{func}', func, result)

    return result


def convert_latex_in_markdown(content):
    """Convert LaTeX formulas (inline and display) within Markdown content.

    Formula detection rules:
    1. $...$ cannot span <td>/<th> cell boundaries (achieved by processing within each cell independently)
    2. $number-style dollar amounts are not treated as formulas and are left unchanged
    3. Genuine formulas are converted to Unicode wherever they appear
    """
    if content is None:
        return content

    result = content
    _counter = [0]
    _display_map = {}

    def apply_latex(text):
        """Run the LaTeX -> Unicode conversion on formula content."""
        return latex_to_unicode(text)

    def is_dollar_amount(inner):
        """Determine whether $...$ content is a dollar amount (plain/comma-separated digits, optional $ prefix)."""
        s = inner.strip()
        # Strip a possible leading $ before checking
        if s.startswith('$'):
            s = s[1:]
        return bool(re.fullmatch(r'-?\d{1,3}(?:,\d{3})*', s))

    # ---- Step 1: extract each cell's content into a placeholder, processing formulas within the cell ----
    # Note: use </(?:td|th)> instead of </\1> to avoid Python re's \1 backreference bug
    cell_map = {}

    def process_formula_in_cell(inner):
        r"""Convert $...$ formulas within a cell.
        (?!\$) prevents $x^2$ from spanning the outer $$x^2$$."""
        _cell_formula_counter = [0]
        _cell_formula_map = {}

        # Step a: handle $$...$$ display formulas first (to avoid mishandling inner $...$)
        def process_display_in_cell(m):
            formula = m.group(1)
            converted = apply_latex(formula)
            key = f'\x00CELLFORMULA{_cell_formula_counter[0]}\x00'
            _cell_formula_counter[0] += 1
            _cell_formula_map[key] = ('$$' + converted + '$$')
            return key
        inner = re.sub(
            r'\$\$((?:[^$]|\\(?!\$))*?)\$\$',
            process_display_in_cell,
            inner, flags=re.DOTALL
        )

        # Step b: then handle $...$ inline formulas
        def repl(m):
            if is_dollar_amount(m.group(1)):
                return m.group(0)
            return apply_latex(m.group(1))
        inner = re.sub(r'\$(.+?)\$(?!\$)', repl, inner)

        # Step c: restore the cell's display formulas
        for key, original in _cell_formula_map.items():
            inner = inner.replace(key, original)
        return inner

    def protect_cell(m):
        key = f'\x00CELL{_counter[0]}\x00'
        _counter[0] += 1
        # group(1) = tag like <td>, group(2) = content, group(3) = </td>
        tag_open = m.group(1)
        inner = m.group(2)
        tag_close = m.group(3)
        processed = tag_open + process_formula_in_cell(inner) + tag_close
        cell_map[key] = processed
        return key

    result = re.sub(
        r'(<(?:td|th)(?:\s[^>]*)?>)(.+?)(</(?:td|th)>)',
        protect_cell, result, flags=re.DOTALL
    )

    # ---- Step 2: replace $$...$$ with placeholders to avoid re-processing inner $...$ ----
    def process_display_math(m):
        formula = m.group(1)
        if is_dollar_amount(formula):
            return m.group(0)
        converted = apply_latex(formula)
        key = f'\x00DISPLAY{_counter[0]}\x00'
        _counter[0] += 1
        _display_map[key] = ('$$' + converted + '$$')
        return key

    result = re.sub(
        r'\$\$((?:[^$\\]|\\(?!\$))+?)\$\$',
        process_display_math, result, flags=re.DOTALL
    )

    # ---- Step 3: handle inline formulas $...$ ----
    def is_latex_formula(inner):
        """Determine whether $...$ content is a genuine LaTeX formula."""
        s = inner.strip()
        # Content with LaTeX-characteristic characters is a formula
        latex_indicators = [
            r'\\',          r'[\^_]',       r'[{}/]',
            r'[≤≥≠≈≡]',    r'[α-ω]',       r'[Α-Ω]',
            r'[∑∏∫√]',    r'[±×÷·]',      r'[→←⇒⇐]',
        ]
        if any(re.search(p, s) for p in latex_indicators):
            return True
        # Pure letters or alphanumeric combinations (math variable names) are treated as formulas
        if re.fullmatch(r'[a-zA-Z][a-zA-Z0-9]*', s):
            return True
        return False

    def process_formula(m):
        inner = m.group(1)
        if is_dollar_amount(inner) or not is_latex_formula(inner):
            return m.group(0)  # dollar amounts or plain text are left unchanged
        return apply_latex(inner)

    # At this point there are no more <td>/<th> in the content, so we can match $...$ directly
    result = re.sub(r'\$([^$\n]+)\$', process_formula, result)

    # ---- Step 4: handle \[...\] and \(...\) ----
    result = re.sub(r'\\\[(.+?)\\\]', process_display_math, result, flags=re.DOTALL)
    result = re.sub(r'\\\((.+?)\\\)', process_formula, result, flags=re.DOTALL)

    # ---- Step 5: restore placeholders ----
    for key, original in _display_map.items():
        result = result.replace(key, original)
    for key, original in cell_map.items():
        result = result.replace(key, original)

    return result


# ===========================================================================
# LaTeX formula normalization
# ===========================================================================

def latex_formula_normalization(text, category=None):
    """Normalize LaTeX formulas in Markdown

    Main handling: merging adjacent formulas
    - For adjacent formulas (inline $...$ or display $$...$$), if the text between
      them consists only of whitespace, spaces, commas, or an "and" separator, merge
      these adjacent formulas into a single formula
    - For the old_scans_math category, also split the \begin{aligned} environment into
      multiple independent single-line formulas

    Args:
        text: Markdown text content
        category: Category name (for future extension of category-specific normalization strategies)

    Returns:
        The normalized text
    """
    if text is None:
        return text

    result = text

    # ---- Merge display formulas: $$...$$ ... $$...$$ ----
    result = _merge_adjacent_display_formulas(result)

    # ---- Merge inline formulas: $...$ ... $...$ ----
    result = _merge_adjacent_inline_formulas(result)

    # ---- For the old_scans_math category, split the aligned environment ----
    if category == "old_scans_math":
        result = _split_aligned_formulas(result)

    return result


def _is_only_separators(content):
    """Check whether the content consists only of separators (whitespace, spaces, commas, and/the "and" keyword)"""
    if content is None:
        return True
    stripped = content.strip()
    if not stripped:
        return True
    lines = content.split('\n')
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if not re.match(r'^(\s*,?\s*(?:and)?\s*)*$', line_stripped, re.IGNORECASE):
            return False
    return True


def _do_merge(formula1, formula2, separator=' '):
    """Merge two formulas into one, preserving the separator"""
    def get_content(f):
        f = f.strip()
        if f.startswith('$$') and f.endswith('$$'):
            return f[2:-2]
        if f.startswith('$') and f.endswith('$'):
            return f[1:-1]
        return f

    content1 = get_content(formula1)
    content2 = get_content(formula2)
    merged_content = content1 + separator + content2

    if formula1.startswith('$$'):
        return '$$' + merged_content + '$$'
    else:
        return '$' + merged_content + '$'


def _merge_adjacent_display_formulas(text):
    """Merge adjacent display formulas $$...$$ ... $$...$$"""
    segments = []
    last_end = 0
    for m in re.finditer(r'\$\$((?:[^$\\]|\\(?!\$))+?)\$\$', text, flags=re.DOTALL):
        if m.start() > last_end:
            before = text[last_end:m.start()]
            if before:
                segments.append(('text', before))
        segments.append(('formula', m.group(0)))
        last_end = m.end()

    if last_end < len(text):
        segments.append(('text', text[last_end:]))

    if len(segments) < 2:
        return text

    formula_count = sum(1 for seg_type, _ in segments if seg_type == 'formula')
    if formula_count < 2:
        return text

    result_parts = []
    i = 0
    while i < len(segments):
        seg_type, seg_val = segments[i]

        if seg_type == 'text':
            result_parts.append(seg_val)
            i += 1
            continue

        merged = seg_val
        j = i + 1
        while j < len(segments):
            next_seg_type, next_seg_val = segments[j]

            if next_seg_type == 'text':
                if _is_only_separators(next_seg_val):
                    if j + 1 < len(segments) and segments[j + 1][0] == 'formula':
                        sep_text = segments[j][1]
                        if not sep_text.strip():
                            sep_text = ' '
                        merged = _do_merge(merged, segments[j + 1][1], sep_text)
                        j += 2
                        continue
                    else:
                        break
                else:
                    break

            if next_seg_type == 'formula':
                merged = _do_merge(merged, next_seg_val, ' ')
                j += 1
                continue

            j += 1

        result_parts.append(merged)
        i = j

    return ''.join(result_parts)


def _merge_adjacent_inline_formulas(text):
    """Merge adjacent inline formulas $...$ ... $...$"""
    segments = []
    last_end = 0

    # Use negative lookahead to ensure we don't match inside $$...$$
    # First replace $$...$$ with placeholders, then restore after processing
    display_placeholders = {}
    ph_counter = [0]

    def replace_display(m):
        key = f'\x00DISPLAY_PH{ph_counter[0]}\x00'
        display_placeholders[key] = m.group(0)
        ph_counter[0] += 1
        return key

    # Protect and restore $$...$$
    processed = re.sub(r'\$\$((?:[^$\\]|\\(?!\$))+?)\$\$', replace_display, text, flags=re.DOTALL)

    for m in re.finditer(r'\$([^$\n]+)\$', processed):
        if m.start() > last_end:
            before = processed[last_end:m.start()]
            if before:
                segments.append(('text', before))

        inner = m.group(1)
        s = inner.strip()
        if re.fullmatch(r'-?\d{1,3}(?:,\d{3})*', s):
            segments.append(('text', m.group(0)))
        else:
            segments.append(('formula', m.group(0)))
        last_end = m.end()

    if last_end < len(processed):
        segments.append(('text', processed[last_end:]))

    if len(segments) < 2:
        # No inline formulas, restore the original text
        for key, val in display_placeholders.items():
            processed = processed.replace(key, val)
        return processed

    formula_count = sum(1 for seg_type, _ in segments if seg_type == 'formula')
    if formula_count < 2:
        for key, val in display_placeholders.items():
            processed = processed.replace(key, val)
        return processed

    result_parts = []
    i = 0
    while i < len(segments):
        seg_type, seg_val = segments[i]

        if seg_type == 'text':
            result_parts.append(seg_val)
            i += 1
            continue

        merged = seg_val
        j = i + 1
        while j < len(segments):
            next_seg_type, next_seg_val = segments[j]

            if next_seg_type == 'text':
                if _is_only_separators(next_seg_val):
                    if j + 1 < len(segments) and segments[j + 1][0] == 'formula':
                        sep_text = segments[j][1]
                        if not sep_text.strip():
                            sep_text = ' '
                        merged = _do_merge(merged, segments[j + 1][1], sep_text)
                        j += 2
                        continue
                    else:
                        break
                else:
                    break

            if next_seg_type == 'formula':
                merged = _do_merge(merged, next_seg_val, ' ')
                j += 1
                continue

            j += 1

        result_parts.append(merged)
        i = j

    result = ''.join(result_parts)

    # Restore $$...$$ placeholders
    for key, val in display_placeholders.items():
        result = result.replace(key, val)

    return result


def _split_aligned_formulas(text):
    r"""Split the \begin{aligned} environment into multiple independent single-line formulas

    Handles the compound case where $$...$$ contains both an aligned environment and
    other formulas:
    1. First separate the aligned and non-aligned parts
    2. Split the aligned part by rows
    3. Keep the non-aligned parts as-is
    4. Wrap all parts in $$...$$

    For example:
        $$\begin{aligned} a \\ b \end{aligned} c$$
    is converted to:
        $$a$$
        $$b$$
        $$c$$
    """
    result_parts = []
    last_end = 0

    for m in re.finditer(r'\$\$((?:[^$\\]|\\(?!\$))+?)\$\$', text, flags=re.DOTALL):
        # Handle the text before $$
        if m.start() > last_end:
            result_parts.append(text[last_end:m.start()])

        content = m.group(1)

        # Find aligned environments
        aligned_pattern = r'\\begin\{aligned\}(.+?)\\end\{aligned\}'
        aligned_matches = list(re.finditer(aligned_pattern, content, re.DOTALL))

        if not aligned_matches:
            # No aligned environment, keep the original formula
            result_parts.append(m.group(0))
        else:
            # Handle mixed content of aligned and non-aligned parts
            offset = 0
            for am in aligned_matches:
                # Add the unprocessed part before the aligned environment (non-aligned formula)
                if am.start() > offset:
                    non_aligned = content[offset:am.start()]
                    if non_aligned.strip():
                        result_parts.append('$$' + non_aligned.strip() + '$$')
                        result_parts.append('\n')

                # Handle the aligned environment
                aligned_content = am.group(1)
                rows = _extract_aligned_rows(aligned_content)

                if len(rows) > 1:
                    # Multiple rows, split
                    for idx, row in enumerate(rows):
                        result_parts.append('$$' + row.strip() + '$$')
                        if idx < len(rows) - 1:
                            result_parts.append('\n')
                        elif am.end() < len(content) and content[am.end():].strip():
                            # There is more content after the last aligned row, need a newline separator
                            result_parts.append('\n')
                elif len(rows) == 1:
                    # Single row, extract content and remove & symbols
                    result_parts.append('$$' + rows[0] + '$$')
                    if am.end() < len(content) and content[am.end():].strip():
                        result_parts.append('\n')
                # else: empty row, output nothing

                offset = am.end()

            # Add the remaining content after the aligned environment (non-aligned formula)
            if offset < len(content):
                remaining = content[offset:]
                if remaining.strip():
                    result_parts.append('$$' + remaining.strip() + '$$')

        last_end = m.end()

    # Handle trailing text
    if last_end < len(text):
        result_parts.append(text[last_end:])

    return ''.join(result_parts)


def _extract_aligned_rows(aligned_content):
    """Extract each row formula from the content of an aligned environment

    Uses \\\\ as the row separator (in LaTeX, \\\\ is the newline command)
    Also removes the & symbols in each row (the alignment markers in an aligned environment)

    Returns:
        list[str]: List of per-row formula content
    """
    # Escaped double-backslash separator
    # \\\\ in the regex matches two backslashes \\, corresponding to LaTeX's \\\\
    rows = re.split(r'\\\\', aligned_content)
    # Filter out empty rows, and remove the & symbols (alignment markers) in each row
    result = []
    for r in rows:
        stripped = r.strip()
        if stripped:
            # Remove & symbols (which may appear at the start, middle, or end)
            cleaned = stripped.replace('&', '')
            result.append(cleaned)
    return result
