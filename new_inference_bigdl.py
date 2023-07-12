from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer
from craynor_utils.utils import RunTracker
from tqdm.auto import tqdm
import torch
import argparse
import time
import json
import os


# parser:
def parse_args(notebook=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", type=str, default="test_inference_bigdl")
    parser.add_argument("--project_root", type=str, default=".")

    parser.add_argument("--model_path", type=str, default="pretrained/chatglm2-6b")
    parser.add_argument("--val_data_path", type=str, default="data/val.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--stats_path", type=str, default="stats.json")
    parser.add_argument("--args_save_path", type=str, default="args.json")

    # get args:
    if notebook:
        return parser.parse_args([])
    else:
        return parser.parse_args()


class QA:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def inference(self, input_str):

        response, history = model.chat(tokenizer, input_str, history=[])  # error here. why?

        output_str = response
        return output_str


# main
args = parse_args(notebook=False)
rt = RunTracker(project_root=args.project_root, run_tag=args.run_tag)

# save args used
with open(os.path.join(rt.save_path, args.args_save_path), "w", encoding="utf8") as f:
    json.dump(args.__dict__, f, indent=4, ensure_ascii=False)

# load model and convert
model_path = args.model_path
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True).float()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
qa = QA(model, tokenizer)
print("model loaded")

with open(args.val_data_path, "r") as f:
    data = json.load(f)
print("data loaded")

# inference
output_path = os.path.join(rt.save_path, args.output_path)
stats_path = os.path.join(rt.save_path, args.stats_path)
# try
# try:
if True:
    wall_time = list()  # each item is a dict, including idx and wall time
    results = list()
    for item in tqdm(data, desc="inference"):
        start = time.time()
        input_str = item["question"]
        output_str = qa.inference(input_str)
        end = time.time()
        wall_time.append({"idx": item["idx"], "wall_time": end - start})
        results.append(
            {
                "idx": item["idx"],
                "question": item["question"],
                "answer": item["answer"],
                "response": output_str,
                "wall_time": end - start,
            }
        )

        # analyse
        wall_time = sorted(wall_time, key=lambda x: x["wall_time"], reverse=True)
        top10 = wall_time[:10]
        total_time = sum([x["wall_time"] for x in wall_time])
        average_time = total_time / len(wall_time)

        print(f"top10: {top10}")
        print(f"total_time: {total_time}")
        print(f"average_time: {average_time}")
# # keyboard interrupt
# except KeyboardInterrupt:
#     print("KeyboardInterrupt")
# finally:
#     # dump data
#     with open(output_path, "w", encoding="utf8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
#     stats = {"top10": top10, "total_time": total_time, "average_time": average_time}
#     with open(stats_path, "w", encoding="utf8") as f:
#         json.dump(stats, f, indent=4, ensure_ascii=False)
