import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lqr.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from lqr.data_handling import ContrastiveBuilder
import csv
import lqr.concepts.con_data_script as utils
from pathlib import Path
import test_con as test
from lqr.concepts.concept_judge import evaluate
from concept_score import get_concept_score
from ppl_from_file import get_ppl_from_csv

import yaml

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def collect_data(model, tokenizer, key, concept):
    dataguy = ContrastiveBuilder(model, tokenizer)
    sen, other = utils.get_target_and_other_sentences('concepts/filtered_sentences.csv', concept)
    
    filename = key + '-' + concept

    path = Path(PICKLE_JAR + filename + '.pkl')
    if path.exists():
        print(f"{filename} already exists")
    else:
        dataguy.collect_data_batch(sen, 200, filename)
        print("done with ", filename)

    filename = key + '-' + 'non' + concept
    path = Path(PICKLE_JAR + filename + '.pkl')
    if path.exists():
        print(f"{filename} already exists")
    else:
        dataguy.collect_data_batch(other, 200, filename)
        print("done with ", filename)

    filename = key + '-' + concept + '_jac'
    path = Path(PICKLE_JAR + filename + '.pkl')
    if path.exists():
        print(f"{filename} already exists")
    else:
        dataguy.collect_jacobians(sen, 50, filename, max_ctx=32)
        print("done with jac")
    
def load_files(key, concept):
    filename = key + '-' + concept
    con = test.load_file(filename)
    
    filename = key + '-non' + concept
    noncon = test.load_file(filename)

    filename = key + '-' + concept + '_jac'
    jac = test.load_file(filename)

    return con["X"]-noncon["X"], jac["A"]


def main():
    models = [
        "google/gemma-2-2b",
        # "meta-llama/Meta-Llama-3-8B",
        # "Qwen/Qwen2.5-3B",
    ]

    concepts = [
        "football",
        # "circus",
        # "church",
        # "dog",
        # "balloon"
    ]

    model_keys = {  
        "google/gemma-2-2b": "gemma-2-2b",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B"

    }
    l_list = [2]

    recompute=True

    for model_name in models:
        model, tokenizer = utils.load_model(model_name, quant=True)
        print(f"MODEL: {model_name}")
        for concept in concepts:
            print(f"CONCEPT: {concept}")
            key = model_keys[model_name]
            # collect_data(model, tokenizer, key, concept)
            X_contr, A = load_files(key, concept)

            num_trials = 100
            output_filename = key + '/' + key + concept + 'out'
            path = Path('./concepts/' + output_filename + '.csv')
            if path.exists() and not recompute:
                print(f"{output_filename} already exists (output)")
            else:
                test.run_trials(
                    model, 
                    tokenizer, 
                    num_trials,
                    A, 
                    X_contr, 
                    l_list, 
                    filename=output_filename
                )
            
            eval_filename = "./concepts/" + key + "/" + key + concept + "0_shot_eval.csv"
            eval_path = Path(eval_filename)
            if eval_path.exists() and not recompute:
                print(f"{eval_filename} already exists (eval)")
            else:
                evaluate(
                    str(path),
                    concept
                )

            score_filename = "./concepts/" + key + "/" + concept + "_score_eval.csv"
            score_path = Path(score_filename)
            if score_path.exists() and not recompute:
                print(f"{score_filename} already exists (score)")
            else:
                get_concept_score(
                    eval_filename, 
                    key, 
                    concept,
                    l_list
                )

            # ppl_filename = "./concepts/" + key + "/" + concept + "_ppl_eval.csv"
            # ppl_path = Path(ppl_filename)
            # if ppl_path.exists() and not recompute:
            #     print(f"{ppl_filename} already exists (ppl)")
            # else:
            #     get_ppl_from_csv(
            #         score_filename, 
            #         key, 
            #         concept,
            #         l_list
            #     )
            

            
            




if __name__ == "__main__":
    print(f"device: {device}")
    main()

