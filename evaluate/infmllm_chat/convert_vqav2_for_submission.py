import os
import argparse
import json

from evaluate.infmllm_chat.m4c_evaluator import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--dst', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    src = args.src
    test_split = args.test
    dst = args.dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1

    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(test_split)]
    split_ids = set([x['question_id'] for x in test_split])

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        if x['question_id'] not in results:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
        else:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': answer_processor(results[x['question_id']])
            })

    with open(dst, 'w') as f:
        json.dump(all_answers, open(dst, 'w'))
