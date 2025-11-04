from pipeline import pipeline
import multiprocessing
from utils.utils import get_args, ensure_work_dirs
import yaml
import json

def split_nums_evenly(num_workers, nums):
    base = nums // num_workers
    arr = [base] * (num_workers - 1)
    arr.append(nums - base * (num_workers - 1))
    return arr

def load_data_from_config(config):
    paths = config['data_paths']

    if paths['text']:
        with open(paths['text'], 'r', encoding='utf-8') as f:
            text = json.load(f)
    else:
        text = []
    

    if paths['image']:
        with open(paths['image'], 'r', encoding='utf-8') as f:
            figure = json.load(f)
    else:
        figure = []

    if paths['table']:
        with open(paths['table'], 'r', encoding='utf-8') as f:
            table = json.load(f)
    else:
        table = []

    if paths['formula']:
        with open(paths['formula'], 'r', encoding='utf-8') as f:
            formula = json.load(f)
    else:
        formula = []

    if paths['title']:
        with open(paths['title'], 'r', encoding='utf-8') as f:
            title = json.load(f)
    else:
        title = []

    return title, table, text, formula, figure

def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


if __name__ == "__main__":
    
    args = get_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    title, table, text, formula, figure = load_data_from_config(config)
    ensure_work_dirs(config)
    
    
    num_workers = config['num_workers']
    nums = config['nums']
    nums_list = split_nums_evenly(num_workers, nums)

    title_chunks   = chunkify(title,   num_workers)
    table_chunks   = chunkify(table,   num_workers)
    text_chunks    = chunkify(text,    num_workers)
    formula_chunks = chunkify(formula, num_workers)
    figure_chunks  = chunkify(figure,  num_workers)

    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=pipeline,
            args=(
                title_chunks[i],
                text_chunks[i],
                table_chunks[i],
                formula_chunks[i],
                figure_chunks[i],
                nums_list[i],
                i
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
