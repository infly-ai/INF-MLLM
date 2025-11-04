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
    parser.add_argument("--check", action="store_true", help="If this parameter is provided, images with bounding boxes will be generated and saved to the path specified by save_path")
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
    chinese_text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return chinese_text



def resize_image(input_path, output_path, scale_factor):
    image = Image.open(input_path)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_image.save(output_path, dpi=(297,190))


def draw_boxes_on_image(img_path, extracted_data, output_dir):
    """
    Draw bounding boxes on an image and save it with a modified filename.

    :param img_path: Path to the original image
    :param extracted_data: Structured data extracted from JSON, containing bounding box info
    :param output_dir: Directory to save the output image with drawn boxes
    """

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw each bounding box in the extracted data
    for item in extracted_data['form']:
        bbox = item['bbox']
        draw.rectangle(bbox, outline='red', width=2)
    
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
        suffix='.tmp', 
        mode='w', 
        encoding='utf-8'
    )

    try:
        # Write JSON data into the temporary file
        with open(temp_file.name, 'w', encoding='utf-8') as json_file:
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


    page_form = {'image': image_path, 'form': []}


    def calculate_area(bbox):
        left, top, right, bottom = bbox
        return (right - left) * (bottom - top)

    header_info = json_data.get('header', [])
    for header_element in header_info:
        element_type = header_element.get('type')
        content = header_element.get('content', '').strip()
        position = header_element.get('position', {})
        bbox = [position['x'], position['y'],
                position['x'] + position['width'],
                position['y'] + position['height']]
        page_form['form'].append({
            'category': 'header',
            'bbox': bbox,
            'area': calculate_area(bbox),
            'text': content
        })
    

    container_elements = json_data.get('containerElements', [])
    for element in container_elements:
        element_type = element.get('type')
        content = element.get('content', '').strip()
        position = element.get('position', {})
        bbox = [position['x'], position['y'],
                position['x'] + position['width'],
                position['y'] + position['height']]
        bbox = [i*scale for i in bbox]
        if calculate_area(bbox)==0:
            continue
        if element_type == 'section_title' and content:
            page_form['form'].append({
                'category': 'title',
                'level': element.get('level'),
                'bbox': bbox,
                'area': calculate_area(bbox),
                'text': content
            })

        elif element_type in p_classes and content:
            is_cross_column = element.get('isCrossColumn', False)
            category = element_type
            page_form['form'].append({
                'category': category,
                'bbox': bbox,
                'area': calculate_area(bbox),
                'text': content
            })
    
        elif element_type == 'figure' and 'src' in element:
            src = element.get('src', '').strip()
            alt = element.get('alt', 'Image').strip()
            page_form['form'].append({
                'category': 'figure',
                'bbox': bbox,
                'area': calculate_area(bbox),
                'text': alt,
                'src': src
            })
            
        elif element_type == 'table':
            src = element.get('src', '').strip()
            alt = element.get('alt', 'Image').strip()  
            page_form['form'].append({
                'category': 'table',
                'bbox': bbox,
                'area': calculate_area(bbox),
                'text': content,
                'src': src
            })
            
        elif element_type == 'list-item' and content:
            category = 'plain_text'
            page_form['form'].append({
                'category': category,
                'bbox': bbox,
                'area': calculate_area(bbox),
                'text': content,
                'flag': "unordered_list"
            })
            
    footnote_info = json_data.get('pageFootnote', [])
    for page_footnote_element in footnote_info:
        element_type = page_footnote_element.get('type')
        content = page_footnote_element.get('content', '').strip()
        position = page_footnote_element.get('position', {})
        bbox = [position['x'], position['y'],
                position['x'] + position['width'],
                position['y'] + position['height']]
        page_form['form'].append({
            'category': 'page_footnote',
            'bbox': bbox,
            'area': calculate_area(bbox),
            'text': content
        })
            
    footer_info = json_data.get('footer', [])
    for header_element in footer_info:
        element_type = header_element.get('type')
        content = header_element.get('content', '').strip()
        position = header_element.get('position', {})
        bbox = [position['x'], position['y'],
                position['x'] + position['width'],
                position['y'] + position['height']]
        page_form['form'].append({
            'category': 'footer',
            'bbox': bbox,
            'area': calculate_area(bbox),
            'text': content
        })

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
    cleaned_str = re.sub(r'[^\u4e00-\u9fff]', '', input_str)
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
        "GB2312",     # Simplified Chinese encoding standard (6763 characters)
        "GBK",        # Extension of GB2312, supports both Simplified & Traditional
        "GB18030",    # Superset of GBK, universal Chinese character support
        "Big5",       # Traditional Chinese encoding (Taiwan, Hong Kong)
        "HKSCS",      # Big5 extension for Hong Kong
        "UTF-8",      # Universal Unicode (modern standard)
        "UTF-16",     # Unicode (2–4 bytes/char)
        "ISO-2022-CN", # Legacy email/stream Chinese encoding using escape sequences
        "EUC-CN",     # GB2312-based extended Chinese encoding for Unix systems
    ]

    # Try reading file using each encoding until successful
    for manner in chinese_encodings:
        try:
            with open(file_path, 'r', encoding=manner) as file:
                raw_data = file.read()
            break
        except:
            continue

    # Clean extracted text to keep only Chinese characters
    text = clean_string(raw_data)
    return text



def get_random_text_snippet(text_iter):
    text = next(text_iter)['text'][0]

    if len(text) < 20:
        snippet = text
    else:
        start_idx = random.randint(0, max(0, len(text) - 30))
        snippet_length = random.randint(20, 30)
        snippet = text[start_idx:start_idx + snippet_length]

    snippet = snippet.rstrip("，,、;；")

    if not snippet.endswith("。") and not snippet.endswith("."):
        snippet += "。"

    return snippet