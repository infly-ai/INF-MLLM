import itertools
import gzip
import json
import re
import yaml
import os
from collections import OrderedDict
import uuid
import random
import importlib
from tqdm import tqdm
from utils.utils import (
    remove_non_chinese_english_characters,
    clean_dictionary_parts,
    split_text_randomly,
    extract_form_from_json,
    draw_boxes_on_image,
    save_data_to_file,
    insert_image_dict_to_paragraph,
    is_image_small,
    remove_non_chinese_english_characters,
    are_cols_equal,
    add_thead_tbody_to_table,
    is_height_greater_than_width,
    ensure_ends_with_punctuation,
    clean_punctuation_at_end,
    add_random_prefix,
    insert_table_data_randomly,
    rows_count,
    add_thead_tbody_to_table,
    get_random_text_snippet,
)
from utils.utils import get_args
from typing import List
from utils.HeaderFooter import produce_header_footer
from utils.Text import produce_multihead_number, produce_simple_number

from utils.LatexUtil import LatexNormalizer, LatexError
from typing import TextIO

latextool = LatexNormalizer()


class RandomCycle:
    def __init__(self, data):
        self.data = data

    def get_random(self):
        return random.choice(self.data)

class GetData:
    
    def __init__(self, title: List[dict], text: List[dict], table: List[dict], formula: List[dict], figure: List[dict], pid: int):
        self.title = title
        self.text = text
        self.table = table
        self.formula = formula
        self.figure = figure

        self.title_iter = itertools.cycle(self.title)
        self.text_iter = itertools.cycle(self.text)
        self.table_iter = itertools.cycle(self.table)
        self.formula_iter = itertools.cycle(self.formula)
        self.figure_iter = itertools.cycle(self.figure)


    def getData(self):
        args = get_args()
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        layout_config = config['layout_config']
        
        module_path = os.path.join(config["work_path"]["template_path"], config["work_path"]["template_get_data"])
        module_name = module_path.replace(os.sep, ".")
        module = importlib.import_module(module_name)
        if not hasattr(module, "get_data"):
            raise ValueError(f"get_data not in {module_name}.py!")
        func = getattr(module, "get_data")
        input_data = func(self, layout_config)

        return input_data
    