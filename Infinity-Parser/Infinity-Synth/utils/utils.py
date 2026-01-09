import re
import random
from PIL import Image, ImageDraw
import os
import yaml
import json
import argparse
import tempfile


def get_args():
    if hasattr(get_args, "_args"):
        return get_args._args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="If this parameter is provided, images with bounding boxes will be generated and saved to the path specified by save_path",
    )
    args = parser.parse_args()
    get_args._args = args
    return args


def ensure_work_dirs(config):
    work_path = config.get("work_path", {})
    html_path = work_path.get("html_path", "")
    save_image_dir = work_path.get("save_image_dir", "")
    output_gt_path = work_path.get("output_gt_path", "")

    dirs_to_check = [
        os.path.dirname(html_path),
        save_image_dir,
        os.path.dirname(output_gt_path),
    ]

    for d in dirs_to_check:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"[Created] {d}")
        else:
            print(f"[Exists]  {d}")


def remove_non_chinese(text):
    chinese_text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return chinese_text


def resize_image(input_path, output_path, scale_factor):
    image = Image.open(input_path)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_image.save(output_path, dpi=(297, 190))


def draw_boxes_on_image(img_path, extracted_data, output_dir):
    """
    Draw bounding boxes on an image and save it with a modified filename.

    :param img_path: Path to the original image
    :param extracted_data: Structured data extracted from JSON, containing bounding box info
    :param output_dir: Directory to save the output image with drawn boxes
    """

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw each bounding box in the extracted data
    for item in extracted_data["form"]:
        bbox = item["bbox"]
        draw.rectangle(bbox, outline="red", width=2)

    # Create output file name by appending "_boxed" to the original name
    base_name = os.path.basename(img_path)
    file_name, file_ext = os.path.splitext(base_name)
    save_path = os.path.join(output_dir, f"{file_name}_boxed{file_ext}")

    # Save the modified image
    img.save(save_path)
    # print(f"Saved image with drawn boxes to {save_path}")


def save_data_to_file(data, file_path):
    """
    Save data to a file in JSON format.
    Uses a temporary file to avoid partial writes or data corruption
    when disk space is low or writing fails.
    """
    # Create a temporary file in the same directory as the target file
    dir_name, base_name = os.path.split(file_path)
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=dir_name,
        prefix=base_name,
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
    )

    try:
        # Write JSON data into the temporary file
        with open(temp_file.name, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        # Atomic rename: overwrite the target file only after successful write
        os.rename(temp_file.name, file_path)
        # print(f"Data saved successfully to {file_path}")
    except Exception as e:
        # On error, print message and delete the temporary file
        print(f"Error writing to file {file_path}: {e}")
        os.remove(temp_file.name)
    finally:
        # Ensure file handle is closed
        temp_file.close()


def insert_image_dict_to_paragraph(row_list, image_dict):
    random_index = random.randint(0, len(row_list))
    row_list.insert(random_index, image_dict)
    return row_list


def extract_form_from_json(image_path, json_data, scale=1):

    p_classes = [
        "table_caption",
        "table_footnote",
        "figure_caption",
        "formula",
        "formula_caption",
        "text",
        "MathJax",
    ]

    extracted_data = []

    page_form = {"image": image_path, "form": []}

    def calculate_area(bbox):
        left, top, right, bottom = bbox
        return (right - left) * (bottom - top)

    header_info = json_data.get("header", [])
    for header_element in header_info:
        element_type = header_element.get("type")
        content = header_element.get("content", "").strip()
        position = header_element.get("position", {})
        bbox = [
            position["x"],
            position["y"],
            position["x"] + position["width"],
            position["y"] + position["height"],
        ]
        page_form["form"].append(
            {
                "category": "header",
                "bbox": bbox,
                "area": calculate_area(bbox),
                "text": content,
            }
        )

    container_elements = json_data.get("containerElements", [])
    for element in container_elements:
        element_type = element.get("type")
        content = element.get("content", "").strip()
        position = element.get("position", {})
        bbox = [
            position["x"],
            position["y"],
            position["x"] + position["width"],
            position["y"] + position["height"],
        ]
        bbox = [i * scale for i in bbox]
        if calculate_area(bbox) == 0:
            continue
        if element_type == "section_title" and content:
            page_form["form"].append(
                {
                    "category": "title",
                    "level": element.get("level"),
                    "bbox": bbox,
                    "area": calculate_area(bbox),
                    "text": content,
                }
            )

        elif element_type in p_classes and content:
            is_cross_column = element.get("isCrossColumn", False)
            category = element_type
            page_form["form"].append(
                {
                    "category": category,
                    "bbox": bbox,
                    "area": calculate_area(bbox),
                    "text": content,
                }
            )

        elif element_type == "figure" and "src" in element:
            src = element.get("src", "").strip()
            alt = element.get("alt", "Image").strip()
            page_form["form"].append(
                {
                    "category": "figure",
                    "bbox": bbox,
                    "area": calculate_area(bbox),
                    "text": alt,
                    "src": src,
                }
            )

        elif element_type == "table":
            src = element.get("src", "").strip()
            alt = element.get("alt", "Image").strip()
            page_form["form"].append(
                {
                    "category": "table",
                    "bbox": bbox,
                    "area": calculate_area(bbox),
                    "text": content,
                    "src": src,
                }
            )

        elif element_type == "list-item" and content:
            category = "plain_text"
            page_form["form"].append(
                {
                    "category": category,
                    "bbox": bbox,
                    "area": calculate_area(bbox),
                    "text": content,
                    "flag": "unordered_list",
                }
            )

    footnote_info = json_data.get("pageFootnote", [])
    for page_footnote_element in footnote_info:
        element_type = page_footnote_element.get("type")
        content = page_footnote_element.get("content", "").strip()
        position = page_footnote_element.get("position", {})
        bbox = [
            position["x"],
            position["y"],
            position["x"] + position["width"],
            position["y"] + position["height"],
        ]
        page_form["form"].append(
            {
                "category": "page_footnote",
                "bbox": bbox,
                "area": calculate_area(bbox),
                "text": content,
            }
        )

    footer_info = json_data.get("footer", [])
    for header_element in footer_info:
        element_type = header_element.get("type")
        content = header_element.get("content", "").strip()
        position = header_element.get("position", {})
        bbox = [
            position["x"],
            position["y"],
            position["x"] + position["width"],
            position["y"] + position["height"],
        ]
        page_form["form"].append(
            {
                "category": "footer",
                "bbox": bbox,
                "area": calculate_area(bbox),
                "text": content,
            }
        )

    extracted_data.append(page_form)
    return page_form


def clean_string(input_str):
    """
    Remove spaces and non-Chinese characters.
    Keep only Chinese characters (no punctuation).

    Args:
        input_str (str): input string to be cleaned
    Returns:
        str: cleaned string containing only Chinese characters
    """
    # Regex: keep only Chinese characters (Unicode range)
    cleaned_str = re.sub(r"[^\u4e00-\u9fff]", "", input_str)
    return cleaned_str


def read_table_text(file_path):
    """
    Read a text file with multiple possible Chinese encodings,
    return content cleaned to only Chinese characters using clean_string().

    Args:
        file_path (str): path to the file to read
    Returns:
        str: cleaned text extracted from file
    """
    # Supported encodings for Chinese text
    chinese_encodings = [
        "GB2312",  # Simplified Chinese encoding standard (6763 characters)
        "GBK",  # Extension of GB2312, supports both Simplified & Traditional
        "GB18030",  # Superset of GBK, universal Chinese character support
        "Big5",  # Traditional Chinese encoding (Taiwan, Hong Kong)
        "HKSCS",  # Big5 extension for Hong Kong
        "UTF-8",  # Universal Unicode (modern standard)
        "UTF-16",  # Unicode (2–4 bytes/char)
        "ISO-2022-CN",  # Legacy email/stream Chinese encoding using escape sequences
        "EUC-CN",  # GB2312-based extended Chinese encoding for Unix systems
    ]

    # Try reading file using each encoding until successful
    for manner in chinese_encodings:
        try:
            with open(file_path, "r", encoding=manner) as file:
                raw_data = file.read()
            break
        except:
            continue

    # Clean extracted text to keep only Chinese characters
    text = clean_string(raw_data)
    return text


def get_random_text_snippet(text_iter):
    text = next(text_iter)["text"][0]

    if len(text) < 20:
        snippet = text
    else:
        start_idx = random.randint(0, max(0, len(text) - 30))
        snippet_length = random.randint(20, 30)
        snippet = text[start_idx : start_idx + snippet_length]

    snippet = snippet.rstrip("，,、;；")

    if not snippet.endswith("。") and not snippet.endswith("."):
        snippet += "。"

    return snippet


def random_hex_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


# 计算颜色亮度
def get_brightness(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    # 计算相对亮度
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


# 生成合适的字体颜色
def generate_font_color(background_color):
    brightness = get_brightness(background_color)
    # 如果背景较亮，则返回较暗的颜色；如果背景较暗，则返回较亮的颜色
    return "#000000" if brightness > 0.5 else "#000000"
    # return '#000000' if brightness > 0.5 else '#FFFFFF'


def get_text_color(hex_color):
    # 将十六进制颜色转换为 RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    # 计算颜色的亮度
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    # 判断是否使用黑色或白色文本
    return "#000000" if brightness > 128 else "#FFFFFF"


def remove_non_chinese_english_characters(text):
    text = text.replace("\n", "")

    # 正则表达式匹配非中文、非英文字符以及保留中英文标点和空格
    pattern = re.compile(
        r"[^\u4e00-\u9fa5a-zA-Z0-9\u3000-\u303f\uff00-\uffef.,!?;:()\[\]{}“”‘’\'\"\-\—\s]"
    )

    # 替换这些字符为空字符串
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def clean_dictionary_parts(parts):
    """清理字典中所有键和值的非中英文字符"""
    cleaned_parts = {}
    for key, value in parts.items():
        cleaned_key = remove_non_chinese_english_characters(key)
        if isinstance(value, str):
            cleaned_value = remove_non_chinese_english_characters(value)
        elif isinstance(value, dict):
            # 如果值也是字典，递归处理
            cleaned_value = clean_dictionary_parts(value)
        else:
            # 对于非字符串的值，直接使用原值
            cleaned_value = value
        cleaned_parts[cleaned_key] = cleaned_value
    return cleaned_parts


def split_text_randomly(text, min_length=200, max_length=400):
    sentence_endings = re.compile(r"[。！？\.\!\?]+")
    paragraphs = []
    last_end = 0
    while last_end < len(text):
        # 随机确定下一个段落的目标长度
        target_length = random.randint(min_length, max_length)
        next_possible_end = last_end + target_length

        if next_possible_end >= len(text):
            # 如果计算的结束位置超出文本长度，直接添加剩余部分
            paragraphs.append(text[last_end:].strip())
            break

        # 寻找最近的句子结束位置
        match = sentence_endings.search(text, next_possible_end)
        if not match:
            # 如果没有找到结束符，直接使用目标长度（这种情况很少见）
            end = next_possible_end
        else:
            # 尽可能使用句子结束符作为分割点
            end = match.end()

        # 添加当前段落
        paragraphs.append(text[last_end:end].strip())
        last_end = end

    return paragraphs


def is_image_small(image_path, width_threshold=100, height_threshold=100):
    """
    读取图片并判断其尺寸是否小于给定的阈值。

    :param image_path: str, 图片的路径。
    :param width_threshold: int, 宽度的阈值。
    :param height_threshold: int, 高度的阈值。
    :return: bool, 如果图片较小则返回 True，否则返回 False。
    """
    try:
        # 打开图像
        with Image.open(image_path) as img:
            # 获取图像的宽度和高度
            width, height = img.size

            # 检查图像是否小于指定的阈值
            if width < width_threshold and height < height_threshold:
                return True
            else:
                return False
    except Exception as e:
        print(f"无法读取图片: {e}")
        return False


def are_cols_equal(html_table):
    """<tr> 之间没有<td>的不是一个有效行，不计算 span"""
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    if not table:
        return False

    rows = table.find_all("tr")
    if not rows:
        return False

    total_rows = 0

    try:
        for row in rows:
            cols = row.find_all(["td", "th"])
            if len(cols) == 0:  # 非有效行 不计算行数
                continue
            total_rows = total_rows + 1

        col_count = [0 for _ in range(total_rows)]

        for i in range(len(rows)):
            cols = rows[i].find_all(["td", "th"])
            if len(cols) == 0:
                continue
            for col in cols:
                col_count[i] = col_count[i] + int(col.get("colspan", 1))

                if i < total_rows - 1:
                    if (
                        int(col.get("rowspan", 1)) > 1
                    ):  # 如果存在跨行 从跨行起下面一行列数加 该单元格跨列数 尾行的跨行不计算
                        c = int(col.get("colspan", 1))
                        for j in range(i + 1, i + int(col.get("rowspan", 1))):
                            col_count[j] = col_count[j] + c
    except Exception:
        return False

    if len(set(col_count)) == 1:
        return True

    else:
        return False


def rows_count(html_table):
    """
    计算总行数
    <tr> 之间没有<td>的不是一个有效行，不计算 span
    """
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    if not table:
        return 0

    rows = table.find_all("tr")
    if not rows:
        return 0

    total_rows = 0

    for row in rows:
        cols = row.find_all(["td", "th"])
        if len(cols) == 0:  # 非有效行 不计算行数
            continue
        total_rows = total_rows + 1

    return total_rows


def add_thead_tbody_to_table(html_table):
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_table, "html.parser")

    # 找到表格
    table = soup.find("table")

    # 提取表头
    header_row = table.find("tr")  # 获取第一行作为表头
    thead = soup.new_tag("thead")  # 创建<thead>标签
    thead.append(header_row)  # 将第一行添加到<thead>中

    # 提取表格主体
    tbody = soup.new_tag("tbody")  # 创建<tbody>标签
    for row in table.find_all("tr"):  # 遍历剩余行
        if row not in thead:
            tbody.append(row)  # 添加行到<tbody>中

    # 清空原表格内容
    table.clear()

    # 添加新的<thead>和<tbody>到表格中
    table.append(thead)
    table.append(tbody)

    # 返回修改后的HTML表格
    return str(soup)


def ensure_ends_with_punctuation(text):

    punctuation_marks = [".", "!", "?", "。", "！", "？"]
    pattern = r"[.!?。！？]$"

    if re.search(pattern, text):
        return text  # 已经有标点符号，直接返回

    random_punctuation = random.choice(punctuation_marks)
    return text + random_punctuation


def clean_punctuation_at_end(text):
    # 定义终止符和非终止符
    terminal_marks = "。！？.!?"
    non_terminal_marks = "，：；,;:"

    # 正则表达式模式：匹配以非终止符和终止符结尾的情况，如 "，。"、",!" 等
    pattern = f"[{non_terminal_marks}]+([{terminal_marks}])$"

    # 使用正则表达式替换，保留最后一个终止符，删除前面的非终止符
    cleaned_text = re.sub(pattern, r"\1", text)

    return cleaned_text


def is_height_greater_than_width(image_path):
    """判断图片高度是否大于宽度."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return height > 0.8 * width
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return True


def add_random_prefix(text):
    # 中文数字表示
    chinese_numbers = [
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
    ]
    # 阿拉伯数字表示
    arabic_numbers = list(range(1, 20))

    # 33% 概率什么都不加
    if random.random() < 0.33:
        return text  # 不加任何前缀，直接返回原始文本

    # 随机选择一种表示形式：中文数字或阿拉伯数字
    use_chinese = random.choice([True, False])

    if use_chinese:
        # 随机选择一个中文数字
        number = random.choice(chinese_numbers)
    else:
        # 随机选择一个阿拉伯数字
        number = random.choice(arabic_numbers)

    # 随机选择“章”或“节”作为前缀
    prefix_type = random.choice(["章", "节"])

    # 生成前缀
    prefix = f"第{number}{prefix_type}"

    # 返回带有前缀的文本
    return f"{prefix} {text}"


import random
from collections import OrderedDict


def insert_table_data_randomly(ordered_dict, table_data):
    """
    将 table_data 中的元素依次随机插入到 ordered_dict 中，保持 table_data 中的相对顺序不变。

    :param table_data: 一个包含单键值对字典的列表
    :param ordered_dict: 目标有序字典（OrderedDict）
    :return: 插入后的有序字典
    """
    # 获取当前 OrderedDict 的键列表
    current_keys = list(ordered_dict.keys())
    insertion_point = 0  # 记录当前插入点的位置

    for element in table_data:
        # 提取 table_data 中的单键值对
        key, value = next(iter(element.items()))

        # 随机生成一个位置来插入，但要保证新元素的位置在当前插入点之后
        insert_pos = random.randint(insertion_point, len(current_keys))

        # 在 current_keys 中插入新的 key
        current_keys.insert(insert_pos, key)

        # 将新的元素插入到 ordered_dict 中
        ordered_dict[key] = value

        # 更新插入点到新插入元素的索引位置 + 1（保持相对顺序）
        insertion_point = insert_pos + 1

        # 将 ordered_dict 重新排序为 current_keys 的顺序
        ordered_dict = OrderedDict((k, ordered_dict[k]) for k in current_keys)

    return ordered_dict


def get_title_color(bg_color):
    """
    根据背景颜色选择合适的字体颜色，确保对比明显且视觉舒适。

    参数:
    bg_color (str): 背景颜色，格式为 '#RRGGBB' 或 '#RGB'，可以带 '#' 或不带。

    返回:
    str: 推荐的字体颜色，格式为 '#RRGGBB'。
    """
    # 去掉 '#' 符号并扩展简写颜色格式（如 #RGB -> #RRGGBB）
    if bg_color.startswith("#"):
        bg_color = bg_color[1:]
    if len(bg_color) == 3:
        bg_color = "".join([c * 2 for c in bg_color])

    # 将十六进制颜色转换为 RGB 值
    r = int(bg_color[0:2], 16)
    g = int(bg_color[2:4], 16)
    b = int(bg_color[4:6], 16)

    # 计算亮度
    def luminance(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * luminance(r) + 0.7152 * luminance(g) + 0.0722 * luminance(b)

    # 选择字体颜色
    # if lum > 0.179:
    #     # 背景较亮，返回暗色调
    #     return '#000000'  # 黑色
    # else:
    #     # 背景较暗，返回亮色调
    #     return '#FFFFFF'  # 白色

    # 生成彩色前景颜色（根据背景色调整）
    new_r = min(255, max(0, r + 50))
    new_g = min(255, max(0, g + 50))
    new_b = min(255, max(0, b + 50))

    return "#{:02x}{:02x}{:02x}".format(new_r, new_g, new_b)


def align_table_columns(html_code):
    # 使用 BeautifulSoup 解析 HTML 代码
    soup = BeautifulSoup(html_code, "html.parser")

    # 获取所有的表格
    tables = soup.find_all("table")

    # 遍历每个表格
    for table in tables:
        # 获取表格中的所有行
        rows = table.find_all("tr")

        # 遍历每一行
        for row in rows:
            cells = row.find_all(["th", "td"])  # 获取表头和表格单元格

            # 为左列单元格添加 'left-align' 类
            if len(cells) > 0:
                cells[0]["class"] = cells[0].get("class", []) + ["left-align"]

            # 为中间列单元格添加 'center-align' 类
            for i in range(1, len(cells) - 1):
                cells[i]["class"] = cells[i].get("class", []) + ["right-align"]

            # 为右列单元格添加 'right-align' 类
            if len(cells) > 1:
                cells[-1]["class"] = cells[-1].get("class", []) + ["right-align"]

    # 返回修改后的 HTML 字符串
    return str(soup)


def generate_random_table():
    # 随机生成行数和列数（3-6之间）
    rows = random.randint(3, 6)
    cols = random.randint(3, 6)

    # 创建HTML表格的起始标签
    html_table = "<table>\n"

    # 生成表格行和列的数据
    for _ in range(rows):
        html_table += "  <tr>\n"
        for _ in range(cols):
            # 随机选择是生成整数还是浮点数
            if random.choice([True, False]):
                num = random.randint(1, 100)  # 随机整数
            else:
                num = round(random.uniform(1, 100), 2)  # 随机浮点数，保留2位小数

            html_table += f"    <td>{num}</td>\n"
        html_table += "  </tr>\n"

    # 添加HTML表格的结束标签
    html_table += "</table>"

    return html_table


import re


def clean_string(input_str):
    """
    删除字符串中的空格、非中英文字符，保留中文字符，不保留标点符号。

    参数:
    input_str (str): 要处理的输入字符串

    返回:
    str: 处理后的字符串，保留中文字符，去除其他字符
    """
    # 正则表达式，保留中文字符和英文字母
    cleaned_str = re.sub(r"[^\u4e00-\u9fff]", "", input_str)

    return cleaned_str


def read_table_text(file_path):
    chinese_encodings = [
        "GB2312",  # 1980年发布的简体中文编码标准，包含6763个汉字。
        "GBK",  # GB2312的扩展，支持21003个汉字，包含简体和繁体。
        "GB18030",  # 2000年发布，支持所有中文字符，同时兼容GBK和GB2312。
        "Big5",  # 繁体中文编码，主要用于台湾和香港，包含13053个汉字。
        "HKSCS",  # Big5的扩展，主要用于香港，增加了额外的繁体中文字符。
        "UTF-8",  # Unicode的可变长度字符编码，支持所有书写系统，逐渐成为通用标准。
        "UTF-16",  # Unicode的另一种字符编码，每个字符占2到4个字节，支持所有书写系统。
        "ISO-2022-CN",  # 一种使用转义序列来表示中文字符的编码方式，主要用于早期电子邮件系统。
        "EUC-CN",  # 一种基于GB2312的扩展编码方式，常用于Unix/Linux系统。
    ]
    for manner in chinese_encodings:
        try:
            with open(file_path, "r", encoding=manner) as file:
                raw_data = file.read()
            break
        except:
            continue
    text = clean_string(raw_data)
    return text


def get_random_text_snippet(text_iter):
    text = next(text_iter)["text"][0]

    # 确保文本长度足够
    if len(text) < 20:
        snippet = text
    else:
        start_idx = random.randint(0, max(0, len(text) - 30))
        snippet_length = random.randint(20, 30)
        snippet = text[start_idx : start_idx + snippet_length]

    # 移除结尾的“，、；；”等符号，但保留“）】》”等符号
    snippet = snippet.rstrip("，,、;；")

    # 如果末尾不是句号，则加上句号
    if not snippet.endswith("。") and not snippet.endswith("."):
        snippet += "。"

    return snippet
