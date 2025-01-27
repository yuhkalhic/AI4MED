import argparse
import copy
from glob import glob
import json
from pprint import pprint
import sys, os
sys.path.append(os.path.abspath("./"))

from os.path import join as pjoin
import uuid


if __name__ == "__main__":
    from filelock import SoftFileLock
    from tqdm import tqdm
    from models.llm_utils import VLLMChatLLM
    from models.reader import Reader
    from utils.mizhi import Errorer, Printer



class RAG:
    def __init__(self, args):
        self.args = args
        self.llm = VLLMChatLLM(args.llm_name)

        if self.args.system == "reader":
            self.reader = Reader(self.llm, self.args)
            if self.args.plan_name != "":
                self.plan_data = json.load(open(f"alog/{self.args.plan_name}/output_all.json"))
        else:
            raise NotImplementedError

    def run(self, question, idx):
        if self.args.system == "reader":
            if self.args.plan_name == "":
                documents = None
            else:
                documents = open(self.plan_data[idx]["pred"]["doc_path"]).read()
            llm_output = self.reader.run(question, documents)
            pred = {"llm_output": llm_output}
        else:
            raise NotImplementedError
        return pred


def change_arg_to_str(args):
    s = ""
    for k in list(args.__dict__.keys()):
        if k == "subset" and args.__dict__[k] < 0:
            continue
        str_v = str(args.__dict__[k]).replace('-Instruct', '').replace('Meta-', '').replace('-INT4', '').replace('-AWQ', '')
        s += f"{k}={str_v},"
    s = s[:-1]
    return s


def get_args(sys_argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--plan_name", type=str, default="")

    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args(sys_argv)
    return args


def work_item(item, idx):
    question = item["question"]
    pred = rag_system.run(question, idx)
    return pred


if __name__ == "__main__":
    args = get_args()
    exp_name = change_arg_to_str(args)
    if args.debug:
        os.popen(f"rm -r {pjoin('alog', exp_name)}").read()
    os.makedirs(pjoin("alog", exp_name), exist_ok=True)
    if len(glob(pjoin("alog", exp_name, "*.log"))) < 3:
        log_path = pjoin("alog", exp_name, f"{uuid.uuid4()}.log")
        sys.stdout = Printer(log_path)
        sys.stderr = Errorer(log_path)

    rag_system = RAG(args)
    ################ work ################
    output_all_path = pjoin("alog", exp_name, "output_all.json")
    lock_path = output_all_path + ".lock"
    with SoftFileLock(lock_path):
        if os.path.exists(output_all_path):
            output_all = json.load(open(output_all_path, encoding="utf-8"))
        else:
            output_all = json.load(open(f"data/data_processed/{args.dataset}.json", encoding="utf-8"))
            with open(output_all_path, "w") as output_fp:
                json.dump(output_all, output_fp, indent=2, ensure_ascii=False)

    assert len(output_all) > 0

    BATCH_SIZE = 3 if args.debug else 20
    while True:
        with SoftFileLock(lock_path):
            output_all = json.load(open(output_all_path, encoding="utf-8"))
            jobs = [example_index for example_index, example in enumerate(output_all) if "pred" not in example][:BATCH_SIZE]
            if len(jobs) == 0:
                jobs = [example_index for example_index, example in enumerate(output_all) if example["pred"] is None][:BATCH_SIZE]
            if len(jobs) == 0:
                print("All jobs are done ~")
                break
            for example_index in jobs:
                output_all[example_index]["pred"] = None
            with open(output_all_path, "w") as output_fp:
                json.dump(output_all, output_fp, indent=2, ensure_ascii=False)

        jobs_output = []
        for example_index in tqdm(jobs):
            pred = work_item(output_all[example_index], example_index)
            jobs_output.append([example_index, pred])
        rag_system.llm.if_print = False

        with SoftFileLock(lock_path):
            output_all = json.load(open(output_all_path, encoding="utf-8"))
            for example_index, pred in jobs_output:
                output_all[example_index]["pred"] = pred
            with open(output_all_path, "w") as output_fp:
                json.dump(output_all, output_fp, indent=2, ensure_ascii=False)

        if args.debug:
            break

