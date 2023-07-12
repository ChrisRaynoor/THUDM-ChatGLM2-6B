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
    # model path and type
    parser.add_argument("--model_path_fp16", type=str, default="pretrained/chatglm2-6b")
    parser.add_argument(
        "--model_path_int4", type=str, default="pretrained/chatglm2-6b-int4"
    )
    parser.add_argument(
        "--model_type", type=str, default="fp16", choices=["fp16", "int4"]
    )
    parser.add_argument("--data_path", type=str, default="data/train_valid.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--stats_path", type=str, default="stats.json")
    parser.add_argument("--args_save_path", type=str, default="args.json")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    rt = RunTracker(project_root=args.project_root, run_tag=args.run_tag)
    # save args used
    with open(
        os.path.join(rt.save_path, args.args_save_path), "w", encoding="utf8"
    ) as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)



    # load Hugging Face Transformers model with INT4 optimizations
    from bigdl.llm.transformers import AutoModelForCausalLM
    model_path = args.model_path_fp16
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
    model = model.eval()
    
    # run the optimized model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer.encode(input_str, ...)
    output_ids = model.generate(input_ids, ...)
    output = tokenizer.batch_decode(output_ids)

    # load different model according to model type
    # there's also an option to load fp16 and then quantize to int8/int4, but it's not implemented here
    # if args.model_type == "fp16":
    #     model_path = args.model_path_fp16
    #     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #     model = AutoModel.from_pretrained(
    #         model_path, trust_remote_code=True, device="cuda"
    #     )
    # elif args.model_type == "int4":
    #     model_path = args.model_path_int4
    #     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #     model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()

    # # === @fastllm
    # # 加入下面这两行，将huggingface模型转换成fastllm模型
    # # is this correct? 似乎原模型必须fp16
    # from fastllm_pytools import llm

    # model = llm.from_hf(
    #     model, tokenizer, dtype=args.model_type
    # )  # dtype支持 "float16", "int8", "int4"
    # # ===

    model = model.eval()
    print("model loaded")

    with open(args.data_path, "r") as f:
        data = json.load(f)
    print("data loaded")

    # inference
    output_path = os.path.join(rt.save_path, args.output_path)
    stats_path = os.path.join(rt.save_path, args.stats_path)
    batch_inference(
        model,
        tokenizer=tokenizer,
        data=data,
        output_path=output_path,
        stats_path=stats_path,
    )


if __name__ == "__main__":
    main()
