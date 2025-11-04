import json
import sys
from tqdm import tqdm
import random
import sys
import os

current_file = os.path.abspath(__file__)  # 当前文件的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_file))  # 上一级目录
sys.path.append(parent_dir)


from utils.LatexUtil import LatexNormalizer, LatexError
from typing import TextIO



latextool = LatexNormalizer()

prompts = [
    "Please convert the document content into Markdown format.",
]

from bs4 import BeautifulSoup

# def html_table_to_markdown(html: str) -> str:
#     soup = BeautifulSoup(html, "html.parser")
#     table = soup.find("table")
#     if table is None:
#         return "No <table> found."

#     def get_cell_text(cell):
#         return cell.get_text(strip=True).replace("|", "\\|")

#     rows = table.find_all("tr")
#     if not rows:
#         return ""

#     # 提取表头
#     header_cells = rows[0].find_all(["th", "td"])
#     header = [get_cell_text(cell) for cell in header_cells]
#     markdown = "| " + " | ".join(header) + " |\n"
#     markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"

#     # 提取后续行
#     for row in rows[1:]:
#         cells = row.find_all(["td", "th"])
#         line = [get_cell_text(cell) for cell in cells]
#         markdown += "| " + " | ".join(line) + " |\n"

#     return markdown

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
    
    # 构建表格矩阵来处理跨行跨列
    matrix = []
    max_cols = 0
    
    # 第一遍：计算最大列数
    for row in rows:
        cells = row.find_all(["td", "th"])
        col_count = 0
        for cell in cells:
            colspan = int(cell.get("colspan", 1))
            col_count += colspan
        max_cols = max(max_cols, col_count)
    
    # 第二遍：构建矩阵
    for row_idx, row in enumerate(rows):
        if row_idx >= len(matrix):
            matrix.append([None] * max_cols)
        
        cells = row.find_all(["td", "th"])
        col_idx = 0
        
        for cell in cells:
            # 找到下一个空的位置
            while col_idx < max_cols and matrix[row_idx][col_idx] is not None:
                col_idx += 1
            
            if col_idx >= max_cols:
                break
                
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            cell_text = get_cell_text(cell)
            
            # 填充当前单元格及其跨越的区域
            for r in range(row_idx, min(row_idx + rowspan, len(rows))):
                # 确保有足够的行
                while len(matrix) <= r:
                    matrix.append([None] * max_cols)
                
                for c in range(col_idx, min(col_idx + colspan, max_cols)):
                    if r == row_idx and c == col_idx:
                        # 主单元格
                        matrix[r][c] = cell_text
                    else:
                        # 跨越区域标记为空字符串
                        matrix[r][c] = ""
            
            col_idx += colspan
    
    # 确保所有行都有相同的列数
    for row in matrix:
        while len(row) < max_cols:
            row.append("")
        # 将None替换为空字符串
        for i in range(len(row)):
            if row[i] is None:
                row[i] = ""
    
    if not matrix:
        return ""
    
    # 生成Markdown表格
    markdown_lines = []
    
    # 表头
    header_line = "| " + " | ".join(matrix[0]) + " |"
    markdown_lines.append(header_line)
    
    # 分隔线
    separator_line = "| " + " | ".join(["---"] * max_cols) + " |"
    markdown_lines.append(separator_line)
    
    # 数据行
    for row in matrix[1:]:
        data_line = "| " + " | ".join(row) + " |"
        markdown_lines.append(data_line)
    
    return "\n".join(markdown_lines)


def form2docparse(datas):
    
    results = []
    for ind, data in tqdm(enumerate(datas)):
        image = data['image']
        res = []
        try:
            for idx, item in enumerate(data['form']):
                if item['category'] == 'title':
                    res.append('#'*item['level'] + ' ' + item['text'])
                elif item['category'] == "formula":
                    res.append("$$" + latextool(item['text']) + "$$")
                elif item['category'] not in ['figure', 'header', 'footer', 'table', "formula"]:
                    res.append(item['text'])
                elif item['category'] == "table":
                    res.append(html_table_to_markdown(item['text']))
            markdown = '\n\n'.join(res)
            results.append({
                'images': [image],
                'conversations': [
                    {
                        'from': 'human',
                        'value': random.choice(prompts)
                    },
                    {
                        'from': 'gpt',
                        'value': f'```markdown\n{markdown}\n```'
                    }
                ]

            })
        except Exception as e:
            continue
        
    return results
        
def load_and_merge_json_files(directory):
    """读取目录下所有 JSON 文件并合并成一个字典列表"""
    merged_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):  # 如果 JSON 是数组形式，直接合并
                    merged_data.extend(data)
                else:  # 如果是单个对象，加入列表
                    merged_data.append(data)
    return merged_data

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_file>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    # 读取并合并目录下所有 JSON 文件
    merged_data = load_and_merge_json_files(input_dir)
    
    # 处理合并后的数据
    result = form2docparse(merged_data)
    
    # 输出结果到文件
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    
    