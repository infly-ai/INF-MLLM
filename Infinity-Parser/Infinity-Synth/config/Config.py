# Config 

import random

class Config:

    text_colors = [
        '#000000',
        "#333333",
        "#222222",
        "#0a0a0a",
        "#003366",
        "#2f4f4f",
        "#483d8b",
        "#4b0082",
        "#2e8b57",
        "#696969",
        "#800000"
    ]
    background_colors = [
        "transparent",
        "#f8f8f8",
        "#fafafa",
        "#f0f0f0",
        "#e0e0e0",
        "#fff8e1",
        "#f0f8ff",
        "#f5f5f5",
        "#f4fff4",
        "#fff0f5",
        "#fffff0"
    ]

    font_styles = ["normal", "italic", "oblique"]


    fonts = {
            "english": [
                "Times New Roman", 
                "Georgia", 
                "Garamond", 
                "Arial", 
                "Helvetica", 
                "Verdana"
            ],
            "chinese": [
                "SimSun",         
                "NSimSun",        
                "SimHei",         
                "Microsoft YaHei",
                "KaiTi",          
                "FangSong"        
            ]
        }


    font_size_options = {
        "title": [ "10pt", "11pt", "12pt", "13pt"],  
        "authors": ["9pt", "10pt", "11pt"],       
        "abstract": ["10pt","8pt", "9pt"],      
        "content": [ "9pt", "10pt", "11pt", "12pt"],          
        "table": ["10px", "9px", "11px", '12px'],
        "width": [155, 160, 165, 170],
        "table_caption": ["10px", "9px", "11px"],
        "container_img_width": [85, 90, 95, 100],
        "abstract_img_width": [85, 90, 95, 100],
        "head_figure_width": [ 60, 70, 80, 90],
        
    }
    
    table = {
        "line_colors": [
        '#000000',
        "#333333", 
        "#222222",  
        "#0a0a0a",  
        "#003366",  
        "#2f4f4f",  
        "#483d8b",  
        "#4b0082",  
        "#2e8b57",  
        "#696969",  
        "#800000"   
    ],
        "back_color": [
        "transparent",
        "#f8f8f8",  
        "#fafafa",  
        "#f0f0f0", 
        "#e0e0e0", 
        "#fff8e1", 
        "#f0f8ff",  
        "#f5f5f5", 
        "#f4fff4", 
        "#fff0f5", 
        "#fffff0"  
    ],
        "align": ['center', 'left'],
        "width": [80, 90, 100],
    
    }

    align = ['center', 'left']
    
    continer = {
        "h3p_gap": ["1px", "3px", "5px", "7px"],
        "column_gap": ["20px", "25px", "30px"],
        "margin_bottom": ["8px", "10px", "12px", "16px"],
        "line_height": [1.5, 1.6, 1.7, 1.8],
        "align": ['center', 'left']
        }
    
    header = {
        "font_size": ["10pt","8pt", "9pt"],
        
        }
    
    footer = {
        "font_size": ["10pt","8pt", "9pt"],
        
    }
    
    container_layout = {
        "left": [60, 62, 64, 66],
        "gap": [1, 2],
        "background_colors": [
            "transparent",
            "#f8f8f8",
            "#fafafa",
            "#f0f0f0",
            "#e0e0e0",
            "#fff8e1",
            "#f0f8ff",
            "#f5f5f5",
            "#f4fff4",
            "#fff0f5",
            "#fffff0"
        ],

        "dark_background_colors": [
            "#2c2c2c",
            "#36454f",
            "#191970",
            "#2f4f4f",
            "#000080",
            "#556b2f",
            "#301934",
            "#800000",
            "#4b0082",
            "#000000"
        ]
    }

    
    page_num = {
        "back_color": [
            "#f8f8f8",
            "#fafafa",
            "#f0f0f0",
            "#e0e0e0",
            "#fff8e1",
            "#f0f8ff",
            "#f5f5f5",
            "#f4fff4",
            "#fff0f5",
            "#fffff0",
            "#000000",
            "#333333",
            "#222222",
            "#0a0a0a",
            "#003366",
            "#2f4f4f",
            "#483d8b",
            "#4b0082",
            "#2e8b57",
            "#696969",
            "#800000"
        ]
    }
    

def random_value_from_list(list_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            list_to_use = getattr(Config, list_name, [])
            if not list_to_use:
                raise ValueError(f"List '{list_name}' not found in Config class.")
            weights = [ 100 if i == 0 else 1 for i in range(len(list_to_use)) ]
            random_value = random.choices(list_to_use, weights=weights, k=1)[0]
            return func(random_value, *args, **kwargs)
        return wrapper
    return decorator


def get_config_value_by_list(list_name):
    @random_value_from_list(list_name)
    def wrapper(random_value):
        return random_value
    return wrapper()

def random_value_from_dict(config_key):
    def decorator(func):
        def wrapper(*args, **kwargs):
            dict_name, key = config_key.split('.')
            config_dict = getattr(Config, dict_name, None)
            if not config_dict:
                raise ValueError(f"Config dictionary '{dict_name}' not found")
            options = config_dict.get(key, [])
            if not options:
                raise ValueError(f"No options available for '{key}' in '{dict_name}'")
            selected_value = random.choice(options)
            return func(selected_value, *args, **kwargs)
        return wrapper
    return decorator


def get_config_value_by_dict(config_key):
    @random_value_from_dict(config_key)
    def wrapper(random_value):
        return random_value
    return wrapper()

def get_config_value(para):
    if len(para.split('.'))>1:
        value = get_config_value_by_dict(para)
        return value
    else:
        return get_config_value_by_list(para)