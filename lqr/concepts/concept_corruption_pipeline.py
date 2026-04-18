import torch as th
# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lqr.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
import csv
import lqr.concepts.con_data_script as utils
from pathlib import Path
import test_con as test
from lqr.concepts.concept_judge import evaluate
from concept_score import get_concept_score
from ppl_from_file import get_ppl_from_csv
import numpy as np
from scipy.linalg import subspace_angles
import pandas as pd

import yaml
import argparse

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def collect_data(model, tokenizer, key, concept):
    dataguy = utils.ContrastiveBuilder(model, tokenizer)
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
    

def random_mat(n, rng=None):
   """
   Generate an n x k matrix
   """
   if rng is None:
       rng = np.random.default_rng()

   X = np.random.randn(n, n)
#    X = X / np.linalg.norm(X, axis=0, keepdims=True)
   return X


def get_target_sentence(csv_path, target):
    sentences = []  # Target concept sentences
    others = {}     # Other concepts: {concept_name: [sentences]}

    # Read CSV and separate sentences
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            concept_name = row["concept"]
            sentence = row["sentence"]

            if concept_name == f"{target}.NOUN":
                sentences.append(sentence)
                break

    return sentences


def topk_svd(mat, k):
    """
    mat: torch.Tensor [out_dim, in_dim]
    returns: U_k, s_k, Vh_k
    """
    # torch.linalg.svd is often faster than numpy and avoids host copies
    U, S, Vh = th.linalg.svd(mat, full_matrices=False)
    return U[:, :k], S[:k], Vh[:k, :]


def spectral_subspace_similarity(U_W, s_W, U_X, s_X, eps=1e-12):
    """
    Spectrally-informed similarity between top-k subspaces.

    Parameters
    ----------
    U_W : (n, k) ndarray
        Top-k left singular vectors of W
    s_W : (k,) ndarray
        Corresponding singular values of W
    U_X : (n, k) ndarray
        Top-k left singular vectors of X
    s_X : (k,) ndarray
        Corresponding singular values of X
    eps : float
        Numerical stability constant

    Returns
    -------
    sim : float in [0, 1]
        Normalized spectral subspace similarity
    """

    # k x k cross-subspace overlap
    C = U_W.T @ U_X

    # form weighted alignment matrix
    A = (s_W[:, None] * C) * s_X[None, :]

    # nuclear norm via SVD
    singvals = np.linalg.svd(A, compute_uv=False)
    nuclear = singvals.sum()

    # Frobenius norms of spectra
    norm_W = np.linalg.norm(s_W)
    norm_X = np.linalg.norm(s_X)

    # normalized similarity
    sim = nuclear / (norm_W * norm_X + eps)
    return sim

def compute_alignment(A, A_corrupted, num_plots=10):

    similarities = []

    k_max = None

    A = A.detach()  # keep on device for SVD
    A_corrupted = A_corrupted.detach()  # keep on device for SVD

    print(f"A shape: {A.shape}")
    print(f"A corrupted shape: {A_corrupted.shape}")

    LATENT_DIM = A.shape[-1]
    k_max = LATENT_DIM // 20
    print(f"k max: {k_max}")

    num_layers = A.shape[0]
    skip = num_layers // num_plots
    

    # initialize per-layer storage
    s_list = []
    U_list = []

    s_list_corrupted = []
    U_list_corrupted = []

    num_layers = len(A)  # assuming all matrices have the same number of layers
    print(f"num layers: {num_layers}")
    skip = num_layers // num_plots

    # ---- process immediately, layer by layer ----
    for layer in range(0, num_layers, skip):
        print(f"Layer {layer}")

        J = A[layer]  # single Jacobian

        # determine k dynamically (matches your logic)
        k = min(J.shape[-1], k_max)

        U_k, s_k, Vh_k = topk_svd(J, k)

        # move minimal data to CPU
        s_list.append(s_k)
        U_list.append(U_k)

        # same computation for corrupted
        Jc = A_corrupted[layer]  # single Jacobian

        # determine k dynamically (matches your logic)
        k = min(Jc.shape[-1], k_max)

        U_k, s_k, Vh_k = topk_svd(Jc, k)

        # move minimal data to CPU
        s_list_corrupted.append(s_k)
        U_list_corrupted.append(U_k)


    for i in range(0, len(s_list)):
    # for i in range(1):
        print(f"Layer {i}")
        
        # Convert all tensors to numpy
        

        # Full SVDs
        
        # w_list = [s / sum(s) for s in s_list]
        w_list = [s / s.sum() for s in s_list]

        # print(len(mats))
        print(U_list[i].shape)
        score = spectral_subspace_similarity(U_list[i], s_list[i], U_list_corrupted[i], s_list_corrupted[i])
        # score = weighted_cos(angles, w_list[j], w_list[r])

        similarities.append(score)

        # print("\n")

    print(f"similarities: {similarities}")

def load_files(key, concept):
    filename = key + '-' + concept
    con = test.load_file(filename)
    
    filename = key + '-non' + concept
    noncon = test.load_file(filename)

    filename = key + '-' + concept + '_jac'
    jac = test.load_file(filename)

    

    return con["X"]-noncon["X"]

def get_jac(model, tokenizer, concept, alpha):

    dataguy = utils.ContrastiveBuilder(model, tokenizer)
    sentence = get_target_sentence('concepts/filtered_sentences.csv', concept)
    _, A_base = dataguy.collect_acts_and_jacs(sentence, 1, 'None', max_ctx=16)
    A = th.zeros_like(A_base[0])
    n = A.shape[-1]
    for i, At in enumerate(A_base[0]):
        A[i] = alpha*At + (1-alpha)*th.from_numpy(random_mat(n)).to(A.device)

    return A, compute_alignment(A_base[0], A)


def main():
    models = {
        "gemma2b": "google/gemma-2-2b",
        "llama8b": "meta-llama/Meta-Llama-3-8B",
        "qwen3b": "Qwen/Qwen2.5-3B",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gemma2b", "qwen3b", "llama8b"],
        default="gemma2b",
    )

    args = parser.parse_args()

    if args.model in models:
        model_name = models[args.model]
        print(f"Running model: {model_name}")
    else:
        raise ValueError("vro...")

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
    l_list = [1.5]

    recompute=True

    alphas = [0, 0.25, 0.5, 0.75, 1]

    print(model_name)

    model, tokenizer = utils.load_model(model_name, quant=True)

    print(f"MODEL: {model_name}")
    for concept in concepts:
        for a in alphas:
            print(f"CONCEPT: {concept}")
            key = model_keys[model_name]
            # collect_data(model, tokenizer, key, concept)
            X_contr = load_files(key, concept)

            A, alignment = get_jac(model, tokenizer, concept, a)

            num_trials = 10
            output_filename = key + '/real_jac' + key + concept + 'out' + f"_alpha{a}_unnormed"
            path = Path('./new_concepts/' + output_filename + '.csv')
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
            print("post run trials")
            
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
                    path, 
                    key, 
                    concept,
                    l_list
                )
            print("post concept score")
            print(f"supposedly saving to : {path}")
            df = pd.read_csv(path)

            #  Add the list as a new column
            df['alignment'] = alignment

            # Save back to CSV (overwrite)
            df.to_csv(path, index=False)
            print(f"saved alignment: {alignment}")

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

