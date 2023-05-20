"""Send a test message."""
import argparse
import json

import requests

from fastchat.model.model_adapter import get_conversation_template


def get_worker_addr(worker_address, controller_addr, model_name):
    if worker_address:
        return worker_address
    else:
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")
        return worker_addr


def test_model_call(worker_addr, model_name, args):
    conv = get_conversation_template(model_name)
    conv.append_message(conv.roles[0], args.message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
    )

    print(f"{conv.roles[0]}: {args.message}")
    print(f"{conv.roles[1]}: ", end="")
    prev = 0
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            output = data["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)
    print("")


def test_swap(args):
    worker_addr = get_worker_addr(args.worker_address, args.controller_address, args.model_name)
    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return
    test_model_call(worker_addr, args.model_name, args)

    # TODO test controller information update: re-get worker_addr
    # swap out
    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model_name": args.model_name,
    }
    response = requests.post(
        worker_addr + "/worker_swap_out",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    # TODO assert invalid call

    # swap in
    gen_params = {
        "model_name": "opt-125m",
        "model_path": "facebook/opt-125m"
    }
    response = requests.post(
        worker_addr + "/worker_swap_in",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    test_model_call(worker_addr, gen_params["model_name"], args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--message", type=str, default="Tell me a story with more than 1000 words."
    )
    args = parser.parse_args()

    test_swap(args)
