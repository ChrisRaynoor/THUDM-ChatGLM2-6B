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
    parser.add_argument("--run_tag", type=str, default="chatglm2-cpp")

    # model config
    parser.add_argument(
        "-m", "--model", default="pretrained/chatglm2-ggml/chatglm2-q4_0-ggml.bin", type=str, help="model path"
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
    results = list()
    analyse_times = list()
    total_time = 0.0
    total_tokens = 0
    total_out_tokens = 0

    try:
        for item in tqdm(data, desc="inference"):
            start = time.time()

            context = item["question"]
            response, len_in, len_out = pipeline.verbose_chat(
                [context],
                max_length=args.max_length,
                max_context_length=args.max_context_length,
                do_sample=args.temp > 0,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temp,
                num_threads=args.threads,
            )

            end = time.time()

            last_result = {
                "idx": item["idx"],
                "question": item["question"],
                "answer": item["answer"],
                "response": response,
                "history": "not returned",
                "wall_time": end - start,
                "len(input_ids)": len_in,
                "len(output_ids)": len_out,
                "token/s": (len_in + len_out) / (end - start),
                "out_token/s": len_out / (end - start),
            }
            results.append(last_result)

            # analyse:
            analyse_start = time.time()

            top10_time = sorted(results, key=lambda x: x["wall_time"])[:10]
            top10_time = [(x["idx"], x["wall_time"]) for x in top10_time]
            total_time += last_result["wall_time"]
            average_time = total_time / len(results)
            total_tokens += last_result["len(input_ids)"] + last_result["len(output_ids)"]
            total_out_tokens += last_result["len(output_ids)"]
            token_ps = total_tokens / total_time
            out_token_ps = total_out_tokens / total_time

            print(f"last_predict: {results[-1]}")
            print("-" * 20 + "stats" + "-" * 20)
            # print(f"top10_time: {top10_time}")
            # print(f"total_time: {total_time}")
            # print(f"average_time: {average_time}")
            # print(f"total_tokens: {total_tokens}")
            # print(f"token_ps: {token_ps}")
            # print(f"total_out_tokens: {total_out_tokens}")
            # print(f"out_token_ps: {out_token_ps}")
            # these abobe are stats, save to a dict:
            stats = {
                "top10_time": top10_time,
                "total_time": total_time,
                "average_time": average_time,
                "average_time": average_time,
                "total_tokens": total_tokens,
                "token_ps": token_ps,
                "total_out_tokens": total_out_tokens,
                "out_token_ps": out_token_ps,
            }
            # print by line:
            for k, v in stats.items():
                print(f"{k}: {v}")
            print("-" * 20 + "stats" + "-" * 20)

            analyse_end = time.time()
            analyse_times.append(analyse_end - analyse_start)

            # print analyse time stats:
            print(f"analyse_time: {analyse_end - analyse_start}")
            print(f"analyse_time_average: {sum(analyse_times) / len(analyse_times)}")
            # print sep line:
            print("=" * 50)

    finally:
        # save and print as utf8, not \u
        with open(output_path, "w", encoding="utf8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        with open(stats_path, "w", encoding="utf8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
