import json
import random
from openai import OpenAI
import os, sys
sys.path.append(os.path.abspath("./"))
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class VLLMChatLLM():
    def __init__(self, llm_name):
        assert "CUDA_VISIBLE_DEVICES" in os.environ
        self.lora_request = LoRARequest(
            lora_name="raft",
            lora_path="/workspace/LLaMA-Factory/saves/llama3-8b/lora/sft/checkpoint-100",
            lora_int_id=0
        )
        self.llm = LLM(
            model=f"{llm_name}",
            enable_prefix_caching=True,
            max_model_len=32000,
            max_num_seqs=3,
            enable_lora=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(f"{llm_name}")
        self.if_print = True

    def run(self, prompt_ls, n_seqs):
        output_ls = []
        text_ls = [self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        ) for prompt in prompt_ls]
        sample_params = SamplingParams(
            n=n_seqs,
            temperature=0,
            stop=[],
            max_tokens=1000,
            seed=0
        )
        response = self.llm.generate(
            prompts=text_ls,
            sampling_params=sample_params,
            use_tqdm=False,
            lora_request=self.lora_request
        )
        for i in range(len(prompt_ls)):
            prompt = prompt_ls[i]
            output = []
            for ii in range(len(response[i].outputs)):
                output.append(response[i].outputs[ii].text)
            if self.if_print:
                print('='*40)
                print(self.tokenizer.decode(self.tokenizer.encode(prompt), skip_special_tokens=True))
                print('-'*40)
                print(output[0])
                print('='*40)
            output_ls.append(output)
        return output_ls


if __name__ == "__main__":
    llm_name = "Qwen2.5-32B-Instruct-AWQ"
    llm = VLLMChatLLM(llm_name="model=Qwen2.5-7B-Instruct,epochs=2,", gpu_studio=0.45)
    print(llm.run(["Tell me a joke, don't use newline"]))