import os
import logging
import logging.handlers
import uuid
from utils.ReadFile import read_files
from utils.utils import extract_form_from_json, draw_boxes_on_image,save_data_to_file, read_table_text
from config.styles import get_styles_num
from core.getData import GetData
from core.Render import Jinja_render, chrome_render 
from utils.table_html import produce_table_html
from utils.utils import get_args
import yaml
from typing import List


def pipeline(title: List[dict], text: List[dict], table: List[dict], formula: List[dict], figure: List[dict], nums: int, process_id: int):
    args = get_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    work_path = config["work_path"]
    html_path = work_path["html_path"].format(i=process_id)
    save_image_dir = work_path["save_image_dir"]
    output_gt_path = work_path['output_gt_path'].format(i=process_id)


    render = chrome_render()
    all_data = []
    data_counter = 0
    total_count = 0


    Input_data = GetData(title, text, table, formula, figure, process_id)
    template_path = work_path["template_path"]
    template = work_path["template_file"]

    while True:
        if len(all_data)>=nums:
            break
        styles = get_styles_num(config)
        input_content = Input_data.getData()

        if input_content is None:
            continue

        Jinja_render(template_path, input_content, template, styles, html_path)

        unique_id = str(uuid.uuid4())
        
        save_image_path = os.path.join(save_image_dir, f"{unique_id}.png")

        cross_column_paragraphs = render.get_location(f"file://{html_path}", save_image_path)
        print(cross_column_paragraphs)
        if cross_column_paragraphs is not None:
            location_info = extract_form_from_json(save_image_path, cross_column_paragraphs)
            all_data.append(location_info)
            data_counter += 1
            total_count += 1
            if args.check:
                os.makedirs(config['defaults']['save_path'], exist_ok=True)
                draw_boxes_on_image(save_image_path, location_info, config['defaults']['save_path'])

            if data_counter >= config['defaults']["save_every_n"]:
                save_data_to_file(all_data, output_gt_path)
                data_counter = 0
            print(f"Process id {process_id}, Acc {total_count}")
    save_data_to_file(all_data, output_gt_path)
    render.close()



