import os
import json
import argparse

def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--prediction", type=str)
    parser.add_argument('-g', "--ground_truth", type=str)
    args = parser.parse_args()


    outputs = {}
    for line_idx, line in enumerate(open(args.prediction)):
        res = json.loads(line)
        question_id = res['question_id']
        text = res['text'].rstrip('.').lower()
        outputs[question_id] = {"questionId": question_id, "answer": text}

    with open(args.ground_truth) as f:
        for line in f.readlines():
            d = json.loads(line)
            outputs[d['question_id']]['annotation'] = d['answer']
    
    r = evaluate_exact_match_accuracy(outputs.values())
    print({'accuracy': r})
