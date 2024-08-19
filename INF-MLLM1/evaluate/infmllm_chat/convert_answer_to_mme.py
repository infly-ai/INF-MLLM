import os,json
import sys,argparse
parser = argparse.ArgumentParser("convert MME Results for Evaluation")

parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--question_file", type=str)
parser.add_argument("--out_path", type=str, required=True)

args = parser.parse_args()
os.makedirs(args.out_path, exist_ok=True)


question_file=args.question_file

question_map={}
with open(question_file) as f:
    for line in f.readlines():
        question_json = json.loads(line)
        question_map[question_json["question_id"]] = question_json

res_map={}
with open(args.answer_file) as f:
    for line in f.readlines():
        answer_json = json.loads(line)
        question_id = answer_json["question_id"]
        try:
            dataset = question_map[question_id]["dataset"]
        except:
            import pdb; pdb.set_trace()
        imagefile = question_map[question_id]["image"]
        question = question_map[question_id]["text"]
        gt = question_map[question_id]["answer"]

        assert answer_json["prompt"] == question
        pred = answer_json["text"]

        res = imagefile+'\t'+repr(question)+'\t'+gt+'\t'+pred
        if dataset not in res_map:
            res_map[dataset] = []
        res_map[dataset].append(res)


mme_datasets = ["OCR", "artwork", "celebrity", "code_reasoning", "color", "commonsense_reasoning", "count", "existence", "landmark", "numerical_calculation", "position",   "posters", "scene", "text_translation"]

for dataset in mme_datasets:
    result_file = open(os.path.join(args.out_path, '{}.txt'.format(dataset)), "w")
    for res in res_map[dataset]:
        result_file.writelines(res+'\n')
    result_file.close()


