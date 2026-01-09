import json
import gzip
from typing import List, Union, Any


def read_files(*file_paths: str) -> List[Union[List[Any], Any]]:
    """
    根据传入的文件路径顺序，读取文件内容，支持 JSON 和 JSONL.GZ 格式。

    :param file_paths: 一个或多个文件路径，支持 .json 和 .jsonl.gz 文件。
    :return: 文件内容的列表，按给定路径顺序返回。
    """
    results = []

    for file_path in file_paths:
        if file_path.endswith(".json"):
            # 读取 JSON 文件
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)

        elif file_path.endswith(".jsonl.gz"):
            # 读取 JSONL.GZ 文件
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
                results.append(data)

        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

    return results
