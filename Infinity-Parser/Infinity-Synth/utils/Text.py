from collections import OrderedDict
import random
import json
from bs4 import BeautifulSoup  # required for check_merged_cells()


def add_html_header(text: str, level: int, serial_num: str) -> str:
    """
    Wrap the given text with an HTML header tag based on level (h2, h3, h4).
    :param text: header text
    :param level: heading level 1–3 (internally mapped to h2–h4)
    :param serial_num: numbering prefix like "1.2.3"
    """
    level = level + 1  # convert 1→h2, 2→h3, 3→h4
    if level not in [2, 3, 4]:
        raise ValueError("Header level must map to h2, h3, or h4")

    return f"<h{level}>{serial_num} {text}</h{level}>"


def generate_next_headings(levels: list, start: str) -> list:
    """
    Given a list of hierarchical levels and a starting heading number,
    generate the subsequent hierarchical numbering.
    Example: levels=[2,3,2], start="2.1" → ["2.1.1", "2.2"]
    """
    current = list(map(int, start.split(".")))
    results = [start]

    for level in levels:
        if level > len(current):
            current.append(1)
        elif level == len(current):
            current[-1] += 1
        else:
            current = current[:level]
            current[-1] += 1

        results.append(".".join(map(str, current)))

    return results[1:]


def generate_random_list(length: int) -> list:
    """
    Generate a random hierarchical list of 1/2/3 levels, where 1 and 3 cannot be adjacent.
    """
    if length <= 0:
        return []

    result = []
    choices = [1, 2, 3]

    for i in range(length):
        if i == 0:
            result.append(random.choice(choices))
        else:
            if result[-1] == 1:
                next_choices = [2]
            elif result[-1] == 3:
                next_choices = [2]
            else:
                next_choices = choices
            result.append(random.choice(next_choices))

    return result


def generate_random_number(level):
    """
    Generate hierarchical numbering based on level depth 1/2/3.
    """
    parts = [random.randint(1, 10) for _ in range(level)]
    return ".".join(map(str, parts))


def produce_multihead_number(text: dict):
    """
    Build multi-level HTML headings and merge adjacent paragraphs randomly.
    """
    level = generate_random_list(len(text))
    start_num = generate_random_number(level[0])
    num_list = generate_next_headings(level, start_num)

    ordered = OrderedDict()
    pre_text = ""

    for i, (key, value) in enumerate(text.items()):
        next_level = level[i + 1] if i + 1 < len(text) else 1
        new_key = add_html_header(key, level[i], num_list[i])

        if next_level > level[i] and random.random() > 0.3 and isinstance(value, str):
            ordered[new_key] = None
            pre_text = value
        else:
            if isinstance(value, dict):
                ordered[new_key] = value
            elif isinstance(value, list):
                value.append(pre_text)
                pre_text = ""
                ordered[new_key] = value
            else:
                ordered[new_key] = value + pre_text
                pre_text = ""

    return ordered


def generate_random_list_only_2(length: int) -> tuple:
    """
    Randomly generate a level list using only {1,2} or {2,3}.
    """
    mode = random.choice(["1,2", "2,3"])
    choices = [1, 2] if mode == "1,2" else [2, 3]
    return random.choices(choices, k=length), mode


def generate_title_numbers(levels, mode):
    """
    Generate hierarchical title numbering, ensuring consistent style per level.
    Reset lower-level counters when higher ones appear.
    """
    if len(levels) > 40:
        print("Too long")
        return []

    counters = {lvl: 1 for lvl in range(1, max(levels) + 1)}
    chinese = [
        "一",
        "二",
        "三",
        "四",
        "五",
        "六",
        "七",
        "八",
        "九",
        "十",
        "十一",
        "十二",
        "十三",
        "十四",
        "十五",
        "十六",
        "十七",
        "十八",
        "十九",
        "二十",
        "二十一",
        "二十二",
        "二十三",
        "二十四",
        "二十五",
        "二十六",
        "二十七",
        "二十八",
        "二十九",
        "三十",
    ]
    chinese_b = [f"（{c}）" for c in chinese]
    arabic = [f"第{x}节" for x in range(1, 51)]

    style_defs = {
        1: [lambda x: chinese_b[x - 1], lambda x: f"第{x}章", lambda x: chinese[x - 1]],
        2: [lambda x: arabic[x - 1], lambda x: f"第{x}节", lambda x: f"（第{x}节）"],
        3: [lambda x: chinese[x - 1], lambda x: chinese_b[x - 1]],
    }

    available_levels = [1, 2] if mode == "1,2" else [2, 3]
    used = set()
    level_styles = {}

    for lvl in available_levels:
        opts = [f for f in style_defs[lvl] if f not in used]
        style = random.choice(opts) if opts else (lambda x: f"{lvl}.{x}")
        level_styles[lvl] = style
        used.add(style)

    result = []
    for lvl in levels:
        if lvl not in available_levels:
            continue
        num = counters[lvl]
        style = level_styles[lvl]
        result.append(style(num))
        counters[lvl] += 1
        for lower in range(lvl + 1, max(levels) + 1):
            counters[lower] = 1

    return result


def produce_simple_number(text: dict):
    """
    Build simple hierarchical headings with either 1–2 or 2–3 rules.
    """
    level, mode = generate_random_list_only_2(len(text))
    num_list = generate_title_numbers(level, mode)

    ordered = OrderedDict()
    pre_text = ""

    for i, (key, value) in enumerate(text.items()):
        next_level = level[i + 1] if i + 1 < len(text) else 1
        new_key = add_html_header(key, level[i], num_list[i])

        if next_level > level[i] and random.random() > 0.3 and isinstance(value, str):
            ordered[new_key] = None
            pre_text = value
        else:
            if isinstance(value, dict):
                ordered[new_key] = value
            elif isinstance(value, list):
                value.append(pre_text)
                pre_text = ""
                ordered[new_key] = value
            else:
                ordered[new_key] = value + pre_text
                pre_text = ""

    return ordered


def check_merged_cells(html_content: str) -> bool:
    """
    Detect if HTML tables contain colspan or rowspan (merged cells).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for table in soup.find_all("table"):
        for cell in table.find_all(["td", "th"]):
            if cell.has_attr("colspan") or cell.has_attr("rowspan"):
                return True
    return False
