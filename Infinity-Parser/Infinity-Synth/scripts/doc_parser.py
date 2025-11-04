import json
import sys
from tqdm import tqdm
import random
import re
import fitz
from pathlib import Path
import concurrent.futures as cf
import multiprocessing
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from bs4 import BeautifulSoup
import re
from utils.utils import get_args


prompts = [
    """
You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

    1. Text Processing:
    - Accurately recognize all text content in the PDF image without guessing or inferring.
    - Convert the recognized text into Markdown format.
    - Maintain the original document structure, including headings, paragraphs, lists, etc.

    2. Mathematical Formula Processing:
    - Convert all mathematical formulas to LaTeX format.
    - Enclose inline formulas with $ $. For example: This is an inline formula $ E = mc^2 $
    - Enclose block formulas with $$ $$. For example: $$ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

    3. Table Processing:
    - Convert tables to Markdown format.

    4. Figure Handling:
    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

    5. Output Format:
    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
    - For complex layouts, try to maintain the original document's structure and format as closely as possible.

    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
    """,
    "请将文档内容转成 Markdown 格式。",
    "请将主要内容转成 Markdown 格式。",
    "请把下面的文档内容转换为结构化的 Markdown 文件，保留所有段落和标题。",
    "请将以下文本转换为 Markdown 格式，保持排版和上下文一致。",
    "请将以下文档内容整理为符合 Markdown 语法的文档。",
    "Please convert the document content into Markdown format.",
    "Please convert the main content into Markdown format.",
    "Please reformat the following text as a structured Markdown document, preserving all headings and paragraphs.",
    "Please transform the content below into a Markdown document, maintaining the original structure and layout.",
    "Please rewrite the following document content as a Markdown file with proper heading hierarchy and context."
]


def html_table_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return "No <table> found."
    
    def get_cell_text(cell):
        return cell.get_text(strip=True).replace("|", "\\|")
    
    rows = table.find_all("tr")
    if not rows:
        return ""
    
    matrix = []
    max_cols = 0
    
    for row in rows:
        cells = row.find_all(["td", "th"])
        col_count = 0
        for cell in cells:
            colspan = int(cell.get("colspan", 1))
            col_count += colspan
        max_cols = max(max_cols, col_count)
    
    for row_idx, row in enumerate(rows):
        if row_idx >= len(matrix):
            matrix.append([None] * max_cols)
        
        cells = row.find_all(["td", "th"])
        col_idx = 0
        
        for cell in cells:
            while col_idx < max_cols and matrix[row_idx][col_idx] is not None:
                col_idx += 1
            
            if col_idx >= max_cols:
                break
                
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            cell_text = get_cell_text(cell)
            
            for r in range(row_idx, min(row_idx + rowspan, len(rows))):
                while len(matrix) <= r:
                    matrix.append([None] * max_cols)
                
                for c in range(col_idx, min(col_idx + colspan, max_cols)):
                    if r == row_idx and c == col_idx:
                        matrix[r][c] = cell_text
                    else:
            
                        matrix[r][c] = ""
            
            col_idx += colspan
    
    for row in matrix:
        while len(row) < max_cols:
            row.append("")
        for i in range(len(row)):
            if row[i] is None:
                row[i] = ""
    
    if not matrix:
        return ""
    
    markdown_lines = []
    
    header_line = "| " + " | ".join(matrix[0]) + " |"
    markdown_lines.append(header_line)
    
    separator_line = "| " + " | ".join(["---"] * max_cols) + " |"
    markdown_lines.append(separator_line)
    
    for row in matrix[1:]:
        data_line = "| " + " | ".join(row) + " |"
        markdown_lines.append(data_line)
    
    return "\n".join(markdown_lines)


def clean_spaces(text: str) -> str:
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def normalize_formula_newlines(text: str) -> str:
    #  $...$ / $$...$$ / \(...\) / \[...\] 
    pattern = re.compile(
        r"(\${1,2}.*?\${1,2}|\\\(.+?\\\)|\\\[.+?\\\])", 
        flags=re.S
    )

    def repl(m):
        content = m.group(0)
        return content.replace("\n", " ")

    return pattern.sub(repl, text)

def render_page_job(item,
                    dpi: int = 180) -> str:
    """
    job = (pdf_path, page_no)  # page_no 从 0 开始

    """
    try:
        img_path = item["image"]
        pdf_path = img_path + ".pdf"

        with fitz.open(pdf_path) as doc:
            pix = doc.load_page(0).get_pixmap(dpi=dpi)
            img_path = Path(img_path)
            if not img_path.exists():
                pix.save(img_path)
            if has_red_cv(img_path):
                return None
    except Exception as e:
        print(e)
        return None

    
def remove_html_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.S)
    
    
def has_red_cv(image_path, ratio=0.001):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    red_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
    return red_ratio > ratio

    
def render_jobs_multiprocess(jobs,
                             dpi = 180,
                             max_workers=40):

    with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(render_page_job, job, dpi=dpi) for job in jobs]
        results = []
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Rendering"):
            if fut.result():
                results.append(fut.result())    
    return results


def form2docparse(datas):
    weights = [9] + [1.0] * (len(prompts) - 1)
    
    results = []
    for ind, data in tqdm(enumerate(datas)):
        image = data['image']
        res = []

        for idx, item in enumerate(data['form']):
            if item['category'] == 'title':
                res.append('#'*item['level'] + ' ' + item['text'])
            elif item['category'] == "table":
                res.append(html_table_to_markdown(item['text']))
            elif item['category'] == 'formula':
                formula = item['text'].replace("<div>", "\n").replace("</div>", "\n").replace("<span>", "\n").replace("</span>", "\n")
                formula = remove_html_comments(formula)
                res.append(formula.strip())
            elif item['category'] not in ['figure', 'header', 'footer', "table", "page_footnote"]:
                res.append(item['text'])
        markdown = '\n\n'.join(res)

        results.append({
            'images': [image],
            'conversations': [
                {
                    'from': 'human',
                    'value': random.choices(prompts, weights=weights, k=1)[0]
                },
                {
                    'from': 'gpt',
                    'value': f'```markdown\n{markdown}\n```'
                }
            ]

        })
        
    return results
        
    
        
if __name__ == "__main__":
    args = get_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    merge_json_files(config["work_path"]["output_gt_path"], os.path.join(os.path.dirname(config["work_path"]["output_gt_path"]), "temp.json"))

    with open( os.path.join(os.path.dirname(config["work_path"]["output_gt_path"]), "temp.json")) as file:
        form_data = json.load(file)
    
    render_jobs_multiprocess(form_data, max_workers=40)
        
    result = form2docparse(form_data)
    
    with open(config["work_path"]["result"], "w") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    
    