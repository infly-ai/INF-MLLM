from config.Config import get_config_value
import random
from utils.utils import get_text_color, random_hex_color , generate_font_color   
import re

def extract_single_number(text):
    match = re.search(r'(\d+)pt', text)
    return int(match.group(1)) if match else None

def produce_stytles():
    page_back_color = get_config_value("page_num.back_color")
    header_back_color = random_hex_color()
    right_backcolor = header_back_color if random.random()>0.4 else random_hex_color()
    
    styles = {
        "incude_image_table": True if random.random()>1 else False,
        
        "title": {
            "font_size": get_config_value('font_size_options.title'),
            "font_family": get_config_value("fonts.chinese"),
            "font_weight": "bold",
            "color": get_config_value('text_colors'),
            "background_color": get_config_value('background_colors'),
            "center": get_config_value("align")
        },
        "authors": {
            "font_size": get_config_value('font_size_options.authors'),
            "font_family": get_config_value("fonts.chinese"),
            "font_weight": "normal",  # Typically, author info is not bold
            "color": get_config_value('text_colors'),
            "background_color": get_config_value('background_colors'),
            "center": get_config_value("align")
        },
        "abstract": {
            "font_size": get_config_value('font_size_options.abstract'),
            "font_family": get_config_value("fonts.chinese"),
            "font_weight": "italic",  # Abstracts are often italicized for emphasis
            "color": get_config_value('text_colors'),
            "background_color": get_config_value('background_colors'),
            "center": get_config_value("align")
        },
        "content": {
            "font_size": get_config_value('font_size_options.content'),
            "font_family": get_config_value("fonts.chinese"),
            "font_weight": "normal",  # Regular content typically does not use bold
            "color": get_config_value('text_colors'),
            "background_color": get_config_value('background_colors')
        },
        "section_title": {
            "font_size": get_config_value('font_size_options.content'),
            "font_family": get_config_value("fonts.chinese"),
            "font_weight": "bold",  # Regular content typically does not use bold
            "color": get_config_value('text_colors'),
            "background_color": get_config_value('background_colors')

        },
        
        "table": {
            
            "font_size": get_config_value('font_size_options.table'),
            "font_family_en": get_config_value("fonts.english"),
            "font_family_zh": get_config_value("fonts.chinese"), 
            # "font_weight": "bold",  # Regular content typically does not use bold
            "line_color": get_config_value('table.line_colors'),
            "background_color": get_config_value('background_colors'),
            "back_color": get_config_value('table.back_color'),
            "align": get_config_value("table.align"),
            "width": get_config_value("table.width"),
            "table_caption": get_config_value("font_size_options.table_caption"),
        },
        "body_text": {
            "font_size": "1em",
            "font_family": "Arial, sans-serif",
            "font_weight": "normal",
            "color": "#444",
            "background_color": "#fff",
            "line_height": "1.6"
        },
        "gap":{
            "h3p_gap":get_config_value("continer.h3p_gap")
        },
        "h3location": get_config_value("continer.align"),
        
        "column_gap": get_config_value("continer.column_gap"),
        
        "title_margin_bottom": get_config_value("continer.margin_bottom"),
        
        "authors_margin_bottom": get_config_value("continer.margin_bottom"),
        
        "abstract_margin_bottom": get_config_value("continer.margin_bottom"),
        
        "abstract_width": get_config_value("font_size_options.width"),
        
        "line_height": get_config_value("continer.line_height"),
        
        "caption":{
            "font_size": get_config_value('font_size_options.content'), 
            "line_height": get_config_value("continer.line_height"),
        },
        "should_cross_column": "True",
        
        "figure_up": "True" if random.random() > 0.5 else None,
        "container_per_width": get_config_value("font_size_options.container_img_width"),
        "abstract_per_width":  get_config_value("font_size_options.abstract_img_width"),
        "head_figure_width": get_config_value("font_size_options.head_figure_width"),
        "three_line": "True" if random.random() > 0.5 else None,
        "two_line": "True" if random.random() > 0.1 else None,
        
        
        "header": {
            "page_num_size":  get_config_value("header.font_size"),
            "background_color": get_config_value('background_colors'),
            
        },

        "footer":{
            "page_num_size": get_config_value("footer.font_size"),
            "background_color": get_config_value('background_colors'),
            
        },
        
        "page_num":{
            "background_color": page_back_color,
            "page_num_coloer": get_text_color(page_back_color)
        },
        
        "container_layout": {
            "left": get_config_value("container_layout.left"),
            "gap": get_config_value("container_layout.gap"),
            "back_color": get_config_value('container_layout.background_colors')
        },
        "header_right": {
            "header_backcolor": header_back_color,
            "right_backcolor": right_backcolor,
            "header_font_color": generate_font_color(header_back_color),
            "right_font_color": generate_font_color(right_backcolor),
            "include_P": "True" if random.random()>0.5 else None,
            "padding_value": random.randint(16,20),
        }
    }
    
    return styles


def get_styles_num(config) -> dict:
    """
    """
    styles = produce_stytles()
    
    styles["columns"] = config["layout_config"]["columns"]
    
    
    return styles
