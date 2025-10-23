import os
import json
import argparse
import torch
import pandas as pd
from utils import load_model_and_tokenizer
from attack import attack
from attack_universal import train_universal_suffix


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Single-prompt attack mode
    single = subparsers.add_parser("single")
    single.add_argument("--dataset_path", default="data/advbench.csv")
    single.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    single.add_argument("--model_family", choices=["llama2", "phi3", "qwen2"], default="llama2")
    single.add_argument("--cache_dir", default=None)
    single.add_argument("--control_length", type=int, default=200)
    single.add_argument("--lr", type=float, default=1e-3)
    single.add_argument("--retry", type=int, default=3)
    single.add_argument("--weight_decay", type=float, default=0.01)
    single.add_argument("--num_steps", type=int, default=20)
    single.add_argument("--max_iterations", type=int, default=15)
    single.add_argument("--output_path", default="data/adv.json")

    # Multi-prompt universal training mode
    multi = subparsers.add_parser("multi")
    multi.add_argument("--dataset_path", default="data/advbench.csv")
    multi.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    multi.add_argument("--cache_dir", default=None)
    multi.add_argument("--control_length", type=int, default=200)
    multi.add_argument("--lr", type=float, default=1e-3)
    multi.add_argument("--weight_decay", type=float, default=0.01)
    multi.add_argument("--num_epochs", type=int, default=100)
    multi.add_argument("--num_steps", type=int, default=25)
    multi.add_argument("--batch_size", type=int, default=10)
    multi.add_argument("--num_train_examples", type=int, default=10)
    multi.add_argument("--num_test_examples", type=int, default=10)
    multi.add_argument("--output_path", default="data/universal.json")

    return parser.parse_args()


def run_single(args):
    df = pd.read_csv(args.dataset_path)
    goals = df["goal"]
    targets = df["target"]
    keywords = df["keyword"]

    model, tokenizer = load_model_and_tokenizer(model_name=args.model_name, cache_dir=args.cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    control_prompt = " ".join(["!"] * args.control_length)

    results_all: list = []
    for fixed_prompt, target_text, keywords_text in zip(goals, targets, keywords):
        fixed_prompt = str(fixed_prompt).strip()
        target_text = str(target_text).strip()
        keywords_list = [k.strip() for k in str(keywords_text).split(',') if k.strip()]

        print(f"\n{'='*50}")
        print(f"ATTACK GOAL: {fixed_prompt}")
        print(f"ATTACK TARGET: {target_text}")
        print(f"{'='*50}")

        attempt = 1
        results = attack(
            model, tokenizer, fixed_prompt, control_prompt, target_text, keywords_list,
            args.num_steps, args.lr, args.weight_decay, args.max_iterations, args.control_length, device,
            model_family=args.model_family,
        )

        while not results[-1]["success"] and attempt < args.retry:
            results = attack(
                model, tokenizer, fixed_prompt, control_prompt, target_text, keywords_list,
                args.num_steps, args.lr, args.weight_decay, args.max_iterations, args.control_length, device,
                model_family=args.model_family,
            )
            attempt += 1

        results_all.append(results)

        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results_all, f)


def run_multi(args):
    df = pd.read_csv(args.dataset_path)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_examples = df_shuffled.iloc[:args.num_train_examples]
    test_examples = train_examples[:args.num_test_examples]

    model, tokenizer = load_model_and_tokenizer(model_name=args.model_name, cache_dir=args.cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    control_prompt = " ".join(["!"] * args.control_length)

    results = train_universal_suffix(
        model, tokenizer, train_examples, test_examples, control_prompt,
        args.num_epochs, args.num_steps, args.batch_size, args.lr, args.weight_decay,
        args.control_length, device,
    )

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f)


def main():
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_multi(args)


if __name__ == "__main__":
    main()