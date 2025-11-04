import random

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


def get_data(self, layout_config):
    
    input_data = {}
    column = []
        
    for element, max_count in layout_config["element"].items():

        insert_count = random.randint(0, max_count)
        if element == "table":
            insert_count = max_count
        if element == 'text':
            insert_count = max_count
        if element == "formula":
            insert_count = max_count


        for _ in range(insert_count):
            if element == "title":
                column.append(next(self.title_iter))
            elif element == "text":
                column.append(next(self.text_iter))
            elif element == "table":
                column.append(next(self.table_iter))
            elif element == "formula":
                formula = next(self.formula_iter)
                #column.append(formula)
                try:
                    formula['latex'] = latextool('$$' + formula['latex'] + '$$')
                except Exception as e:
                    continue
                column.append(formula)
            elif element == "figure":
                column.append(next(self.figure_iter))
            elif element == "page_footnote":
                input_data['page_footnote'] = get_random_text_snippet(self.text_iter)

    random.shuffle(column)

    input_data['body'] = column
    if len(column)<2:
        return None 

    title = None

    for dat in column:
        if dat['type']=="Body":
            title = dat['heading']

    if title is not None:
        head_foot = produce_header_footer( title )
        input_data['header'] = head_foot.get('header', None)
        input_data['footer'] = head_foot.get('footer', None)
        
    return input_data