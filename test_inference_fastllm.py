"""test inference time"""

# load data
import argparse
import json
import os
import time

from craynor_utils.utils import RunTracker
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer  # huggingface transformer 接口


def batch_inference(model, tokenizer, data, output_path, stats_path):
    """inference and analyse time.

    it should record a wall time for each inference, output top 10 time and there index, average time and total time.
    also output a json file, including input data, output and wall time for each inference.

    data is a list. each item like:
    {
    "idx": 0,
    "question": "mNGS和tNGS，两个方法有什么区别？",
    "answer": "目前临床上使用的宏基...
    }
    """
    wall_time = list()  # each item is a dict, including idx and wall time
    results = list()
    for item in tqdm(data, desc="inference"):
        start = time.time()
        # inference code here
        response, history = model.chat(tokenizer, item["question"], history=[])
        end = time.time()
        wall_time.append({"idx": item["idx"], "wall_time": end - start})
        results.append(
            {
                "idx": item["idx"],
                "question": item["question"],
                "answer": item["answer"],
                "response": response,
                "history": history,
                "wall_time": end - start,
            }
        )

        # analyse
        wall_time = sorted(wall_time, key=lambda x: x["wall_time"], reverse=True)
        top10 = wall_time[:10]
        total_time = sum([x["wall_time"] for x in wall_time])
        average_time = total_time / len(wall_time)

        # save and print as utf8, not \u
        with open(output_path, "w", encoding="utf8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        stats = {"top10": top10, "total_time": total_time, "average_time": average_time}
        with open(stats_path, "w", encoding="utf8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"top10: {top10}")
        print(f"total_time: {total_time}")
        print(f"average_time: {average_time}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--run_tag", type=str, default="test_inference")

    # all path should be relative to project root
    parser.add_argument("-m", "--model", default="pretrained/chatglm2-6b", type=str, help="model path")

    parser.add_argument("--origin_model_type", type=str, default="fp16")  # for fastllm
    parser.add_argument(
        "--model_type", type=str, default="int4", choices=["fp16", "int4"]
    )  # model type used for inference

    # input config
    parser.add_argument("--data_path", type=str, default="data/train_valid.json")

    # output config
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--stats_path", type=str, default="stats.json")
    parser.add_argument("--args_save_path", type=str, default="args.json")

    args = parser.parse_args()
    return args


def main():
    # common
    args = get_args()
    rt = RunTracker(project_root=args.project_root, run_tag=args.run_tag)
    with open(os.path.join(rt.save_path, args.args_save_path), "w", encoding="utf8") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)

    # load fp16 model
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = model.eval()
    # convert to fastllm
    from fastllm_pytools import llm

    model = llm.from_hf(model, tokenizer, dtype=args.model_type)  # dtype支持 "float16", "int8", "int4"
    print("model loaded")

    # load data:
    with open(args.data_path, "r") as f:
        data = json.load(f)
    print("data loaded")

    # # inference legacy
    # output_path = os.path.join(rt.save_path, args.output_path)
    # stats_path = os.path.join(rt.save_path, args.stats_path)
    # response, history = model.chat(tokenizer, item["question"], history=[])
    
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
            response, history = model.chat(tokenizer, item["question"], history=[])
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
