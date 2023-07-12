"""test inference time"""
import argparse
from pathlib import Path
import chatglm_cpp
import json
import os
import time

from craynor_utils.utils import RunTracker
from tqdm.auto import tqdm
# from transformers import AutoModel, AutoTokenizer  # huggingface transformer 接口, not for chatglm-cpp


def get_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--run_tag", type=str, default="test_inference")

    # model config
    parser.add_argument(
        "-m", "--model", default="pretrained/chatglm2-ggml/chatglm2-q4_0-ggml.bin", type=Path, help="model path"
    )
    parser.add_argument(
        "-l", "--max_length", default=2048, type=int, help="max total length including prompt and output"
    )
    parser.add_argument("-c", "--max_context_length", default=512, type=int, help="max context length")
    parser.add_argument("--top_k", default=0, type=int, help="top-k sampling")
    parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
    parser.add_argument("--temp", default=0.95, type=float, help="temperature")
    parser.add_argument("-t", "--threads", default=2, type=int, help="number of threads for inference")

    # input config
    parser.add_argument("--data_path", type=str, default="data/train_valid.json")

    # output config
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--stats_path", type=str, default="stats.json")
    parser.add_argument("--args_save_path", type=str, default="args.json")

    args = parser.parse_args()
    return args


def main():
    # common:
    args = get_args()
    rt = RunTracker(project_root=args.project_root, run_tag=args.run_tag)
    with open(os.path.join(rt.save_path, args.args_save_path), "w", encoding="utf8") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)

    # load model:
    pipeline = chatglm_cpp.Pipeline(args.model)
    print("model loaded")

    # load data:
    with open(args.data_path, "r") as f:
        data = json.load(f)
    print("data loaded")

    # inference
    output_path = os.path.join(rt.save_path, args.output_path)
    stats_path = os.path.join(rt.save_path, args.stats_path)
    wall_time = list()  # each item is a dict, including idx and wall time
    results = list()
    try:
        for item in tqdm(data, desc="inference"):
            start = time.time()

            context = item["question"]
            response = pipeline.chat(
                [context],
                max_length=args.max_length,
                max_context_length=args.max_context_length,
                do_sample=args.temp > 0,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temp,
            )

            end = time.time()
            wall_time.append({"idx": item["idx"], "wall_time": end - start})
            results.append(
                {
                    "idx": item["idx"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "response": response,
                    "history": "not returned",
                    "wall_time": end - start,
                }
            )

            # analyse:
            wall_time = sorted(wall_time, key=lambda x: x["wall_time"], reverse=True)
            top10 = wall_time[:10]  # ?
            total_time = sum([x["wall_time"] for x in wall_time])
            average_time = total_time / len(wall_time)

            print(f"top10: {top10}")
            print(f"total_time: {total_time}")
            print(f"average_time: {average_time}")
    finally:
        # save and print as utf8, not \u
        with open(output_path, "w", encoding="utf8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        stats = {"top10": top10, "total_time": total_time, "average_time": average_time}
        with open(stats_path, "w", encoding="utf8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
