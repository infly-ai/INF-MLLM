import random
from typing import List, Dict, Optional, Any


def generate_random_page_num(probability: float = 0.5) -> str:
    """
    Randomly generate a page number HTML block (1–1200).
    With a given probability, use class 'circle-background', otherwise 'page-num'.

    Args:
        probability (float): Probability to select class 'circle-background'. Must be between 0 and 1.

    Returns:
        str: HTML string containing a random page number div.
    """

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    class_name = "circle-background" if random.random() < probability else "page-num"
    page_number = random.randint(1, 1200)

    return f'<div class="{class_name}">{page_number}</div>'


def fill_strings_into_dicts(
    strings: List[str],
    single_string: Optional[str] = None,
    specific_string: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Randomly place strings into header/footer regions.

    Header & Footer each contain: left, mid, right (and optionally "line").
    - `single_string`: placed ONLY in header, random position.
    - `specific_string`: placed in header (random pos) AND MAY also fill header/footer right.

    Args:
        strings (List[str]): List of strings to distribute.
        single_string (str, optional): String forced into header.
        specific_string (str, optional): Special string inserted into header and maybe footer.

    Returns:
        dict: structure like:
        {
            "header": {"left": "", "mid": "", "right": "", "line": "line"},
            "footer": {"left": "", "mid": "", "right": "", "line": "line"}
        }
    """

    result = {
        "header": {"left": None, "mid": None, "right": None},
        "footer": {"left": None, "mid": None, "right": None},
    }

    available_positions = {
        "header": ["left", "mid", "right"],
        "footer": ["left", "mid", "right"],
    }

    # Place single_string only in header
    if single_string:
        pos = random.choice(available_positions["header"])
        result["header"][pos] = single_string
        available_positions["header"].remove(pos)

    # Place specific_string into header + maybe right positions
    if specific_string and available_positions["header"]:
        pos = random.choice(available_positions["header"])
        result["header"][pos] = specific_string
        available_positions["header"].remove(pos)

        # Randomly also put into right of header/footer if possible
        if result["header"]["right"] is None and result["footer"]["right"] is None:
            chosen_dict = random.choice(["header", "footer"])
            result[chosen_dict]["right"] = specific_string
            if "right" in available_positions[chosen_dict]:
                available_positions[chosen_dict].remove("right")

    # Fill remaining strings randomly
    for string in strings:
        chosen_dict = random.choice(["header", "footer"])
        if not available_positions[chosen_dict]:
            chosen_dict = "footer" if chosen_dict == "header" else "header"

        if available_positions[chosen_dict]:
            pos = random.choice(available_positions[chosen_dict])
            result[chosen_dict][pos] = string
            available_positions[chosen_dict].remove(pos)

    # Randomly add separator lines
    if random.random() > 0.5:
        result["header"]["line"] = "line"
    if random.random() > 0.5:
        result["footer"]["line"] = "line"

    return result


def produce_header_footer(text: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Generate a random header/footer structure for a document.

    Args:
        text (str, optional): Title text to sometimes be used as header content.

    Returns:
        dict: header/footer dict with random placement of title, page number and shapes.
    """

    page_num_html = generate_random_page_num(0.3)
    rectangle_html = '<div class="rectangle"></div>' if random.random() > 0.1 else None
    title = text if random.random() > 0.5 else None

    return fill_strings_into_dicts(
        strings=[page_num_html], single_string=title, specific_string=rectangle_html
    )
