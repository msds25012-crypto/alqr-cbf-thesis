import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from steering import LQRSteering
from data_handling import ContrastiveBuilder
import yaml
import random
import json
from linearization import compute_lin_err as OLcompute_lin_err
import matplotlib.pyplot as plt
import numpy as np

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"device: {device}")


def load_model(model_name, quant=False):

    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,          # or load_in_8bit=True
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=th.float16,
            bnb_4bit_quant_type="nf4",  # best for LLMs
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, dtype=th.float32, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
        

def get_safe_prompts():
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset['train'][:]["instruction"]

def load_file(filename):
    try:
        with open(PICKLE_JAR + filename + ".pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def compute_lin_err(nom_acts, jacs, acts, var, cov):
    l1norm_byvar_list = []
    avg_l1norm_byvar_list = []
    l2norm_byvar_list = []
    l2norm_list = []
    linfty_by_var_list = []
    cos_sim_list = []
    err_list = []
    for i in range(nom_acts.shape[0]-1):
        delta_xt = nom_acts[i] - acts[i]
        Adelta_xt = jacs[i] @ delta_xt
        delta_xtp1 = nom_acts[i+1] - acts[i+1]

        err = Adelta_xt - delta_xtp1

        l1byvar = th.sum(th.abs(err / var[i]))

        l1norm_byvar_list.append((l1byvar).item())
        avg_l1norm_byvar_list.append((l1byvar / err.shape[-1]).item())
        l2norm_byvar_list.append(th.norm(err / var[i]).item())
        cos_sim_list.append((th.dot(Adelta_xt, delta_xtp1)/(th.norm(Adelta_xt)*th.norm(delta_xtp1))).item()) # cosine similarity
        l2norm_list.append(th.norm(err).item())
        linfty_by_var_list.append(th.max(th.abs(err)).item())
        # err_list.append(err.tolist())

        # print((th.sum(th.abs(err / var[i]))/err.shape[-1]).item())
        # print((th.norm(err / var[i])).item())
        # print((th.dot(Adelta_xt, delta_xtp1)/(th.norm(Adelta_xt)*th.norm(delta_xtp1))).item()) # cosine similarity
        # print((th.norm(err) / (th.norm(nom_acts[i+1]))).item())
        # print((th.norm(err) / (nom_acts.shape[-1])).item())
        # print(f"max: {th.max(err).item()}, min: {th.min(err).item()}")
    
    data = {
        "l1_by_var": l1norm_byvar_list,
        "avg_l1_by_var": avg_l1norm_byvar_list,
        "l2_by_var": l2norm_byvar_list,
        "l2": l2norm_list,
        "linfty_by_var_list": linfty_by_var_list,
        "cos_sim_list": cos_sim_list,
        # "err_list": err_list
            }
    return data


def helper(X1, X2):
    print(X1.shape)
    print(X2.shape)
    # for i in range(X1.shape[0]):

def lin_err_CL(nom_acts, jacs, acts, U, var, diffs, cov, P):
    # print(nom_acts.shape)
    # print(jacs.shape)
    # print(acts.shape)
    # print(U.shape)
    l1norm_byvar_list = []
    avg_l1norm_bydiffs_list = []
    avg_l1norm_byvar_list = []
    l2norm_byvar_list = []
    l2norm_list = []
    linfty_by_var_list = []
    cos_sim_list = []
    # err_list = []
    l1_list = []
    ratio_list = []
    err_list = []
    
    Pnorm_list = []

    for i in range(nom_acts.shape[0]-1):
        delta_xt_cl = acts[i] - nom_acts[i]
        delta_xt_cl = jacs[i] @ delta_xt_cl + U[i]

        true_delta = acts[i+1] - nom_acts[i+1]
        err = delta_xt_cl - true_delta
        
        
        std = th.sqrt(var[i])

        ratio_list.append((th.norm(err) / th.norm(acts[i] - nom_acts[i])).item())
        # ratio_list.append((th.norm(err) / th.norm(true_delta)).item())


        l1 = th.sum(th.abs(err))
        l1_list.append(l1)

        l1byvar = th.sum(th.abs(err / std))
        l1bydiff = th.sum(th.abs(err / diffs[i+1]))

        l1norm_byvar_list.append((l1byvar).item())
        avg_l1norm_byvar_list.append((l1byvar / err.shape[-1]).item())
        avg_l1norm_bydiffs_list.append((l1bydiff / err.shape[-1]).item())

        l2norm_byvar_list.append(th.norm(err / std).item())
        cos_sim_list.append((th.dot(delta_xt_cl, true_delta)/(th.norm(delta_xt_cl)*th.norm(true_delta))).item()) # cosine similarity
        l2norm_list.append(th.norm(err).item())
        err_list.append(err.tolist())


        errbydiff = err / diffs[i+1]
        # errbyvar = err / var[i]
        Pnorm_list.append(th.sqrt(err.T @ P[i+1] @ err).item())
        # Pnorm_list.append(th.sqrt(errbydiff.T @ P[i+1] @ errbydiff).item())
        # Pnorm_list.append(th.sqrt(errbyvar.T @ P[i+1] @ errbyvar).item())
        # linfty_by_var_list.append(th.max(th.abs(err)).item())
        # l1byvar = th.sum(th.abs(err / var[i]))

        # l1norm_byvar_list.append((l1byvar).item())
        # avg_l1norm_byvar_list.append((l1byvar / err.shape[-1]).item())
        # l2norm_byvar_list.append(th.norm(err / var[i]).item())
        # cos_sim_list.append((th.dot(delta_xt_cl, true_delta)/(th.norm(delta_xt_cl)*th.norm(true_delta))).item()) # cosine similarity
        # l2norm_list.append(th.norm(err).item())
        linfty_by_var_list.append(th.max(th.abs(err)/ std).item())
        # err_list.append(err.tolist())
        # print((th.sum(th.abs(err / var[i]))/err.shape[-1]).item())
        # print((th.dot(delta_xt, true_delta)/(th.norm(delta_xt)*th.norm(true_delta))).item()) # cosine similarity

    data = {
        "l1": l1_list,
        "l1_by_var": l1norm_byvar_list,
        "avg_l1_by_var": avg_l1norm_byvar_list,
        "avg_l1_by_diffs": avg_l1norm_bydiffs_list,
        "ratio": ratio_list,
        "l2_by_var": l2norm_byvar_list,
        "l2": l2norm_list,
        "linfty_by_var_list": linfty_by_var_list,
        "cos_sim_list": cos_sim_list,
        "err_list": err_list,
        "Pnorm_list": Pnorm_list
            }
    return data


def solve_backwards_lyapunov(A):
    N = A.shape[0]
    n = A.shape[1]

    Q = th.eye(n, device=device)
    P_T = th.eye(n, device=device)
    
    P = th.zeros((N+1, n, n),device=device)
    P[N] = P_T

    A_gpu = A.to(device)
    for k in reversed(range(N)):
        P[k] = A_gpu[k].T @ P[k+1] @ A_gpu[k] + Q

    return P.detach().cpu()

def op_norm(A, Pi, Pip1):
    L = th.linalg.cholesky(Pi)

    M = A.T @ Pip1 @ A

    X = th.cholesky_solve(M,L)
    lambda_max = th.linalg.eigvalsh(X).max()

    return th.sqrt(lambda_max).item()


def is_contracting(A, Pi, Pip1, gamma=1.0):
    lhs = A.T @ Pip1 @ A
    rhs = gamma**2 * Pi
    eigs = th.linalg.eigvalsh(lhs-rhs)
    return eigs.max() <= 0

def bound_at_k(k, delta_x0, errs, A_cl, P):
    phi_k_0 = th.eye(A_cl.shape[-1], device=device)
    # phi_bar = 1

    sigma = 0


    for i in range(k-1):
        phi_k_0 = A_cl[i] @ phi_k_0
        # phi_bar *= op_norm(A_cl[i], P[i], P[i+1])

    # delta_xk = delta_x0

    t1 = op_norm(phi_k_0, P[0], P[k]) * th.sqrt(delta_x0.T @ P[0] @ delta_x0)
    # t1 = phi_bar


    for i in range(k):
        phi_k_i = th.eye(A_cl.shape[-1], device=device)
        # sigma += op_norm(A_cl[i], P[i], P[i+1])*th.sqrt(errs[i].T @ P[i] @ errs[i])
        for j in range(i+1, k):
            phi_k_i = A_cl[j] @ phi_k_i
        

        # delta_xk = A_cl[i] @ delta_xk
        # sigma += op_norm(A_cl[i], P[i], P[i+1])*errs[i] / th.sqrt(delta_xk.T @ P[i] @ delta_xk)
        # print(f"P[i+1] shape {P[i+1].shape}")
        # print(f"P[i+1] shape {P[k].shape}")
        # print(f"phi k i] shape {P[k].shape}")
        sigma += op_norm(phi_k_i, P[i+1], P[k])*errs[i]
    

    bound = t1 + sigma

    return bound



def main():
    model_name = "google/gemma-2-2b"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    # model_name = "Qwen/Qwen2.5-3B"
    # model_name = "meta-llama/Meta-Llama-3-8B"

    # model_name = "google/gemma-2-9b-it"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"


    model, tokenizer = load_model(model_name, quant=True)
    # messages = [
    #     {"role": "user", "content": p} for p in prompt
    # ]
    # print(messages)


    prompts = get_safe_prompts()
    rand_prompts = []
    for p in prompts:
        ind = random.randint(0, len(p))
        rand_prompts.append(p[:ind])
    print(rand_prompts[0])


    noms = []


    steer = LQRSteering(model, tokenizer, 10, 0.01, 100)
    for i in range(3):    
        out = steer.track_tokens(rand_prompts[0], rand_prompts[1+i], 1)
        X = steer.X[0][:,-1,:].detach().cpu()
        X_cl = steer.X_cl[0][:,-1,:].detach().cpu()
        A = steer.A.detach().cpu()
        K = steer.K.detach().cpu()
        U = steer.U.detach().cpu()
        dat = {
            "X_nom": X,
            "A_nom": A,
            "X_cl": X_cl,
            "K": K,
            "U": U,
        }
        noms.append(dat)
        print(f"Steered out: {out}")

    

    dataguy = ContrastiveBuilder(model, tokenizer)

    acts_new = dataguy.collect_activations(prompts=prompts, num_samples=1000)

    cov_mat = [th.cov(acts_new[:,i,:].T) for i in range(1, acts_new.shape[1])]
    variances = [th.diag(th.cov(acts_new[:,i,:].T)) for i in range(1, acts_new.shape[1])]

    diffs = th.zeros_like(acts_new[0])
    for i in range(acts_new.shape[1]):
        for j in range(acts_new.shape[2]):
            min = th.min(acts_new[:,i,j])
            max = th.max(acts_new[:,i,j])
            diffs[i,j] = max-min

    # single_step_OL = OLcompute_lin_err(steer.X[0][:,-1,:].detach().cpu(), steer.A.detach().cpu(), acts_new[0], variances, diffs)
    
    dat=noms[0]
    X_nom_cpu = dat["X_nom"]
    
    print("shapes")
    print(X_nom_cpu.shape)
    print(X_cl.shape)
    print("end shapes")

    # print("\nOL l1:", single_step_OL["l1"][0])
    # print("\nOL avg_l1_by_var:", single_step_OL["avg_l1_by_var"][0])
    # print("\nOL avg_l1_by_diffs:", single_step_OL["avg_l1_by_diffs"][0])
    # print("\nOL l1_by_var:", single_step_OL["l1_by_var"][0])
    # print("\nOL l2_by_var:", single_step_OL["l2_by_var"][0])
    # print("\nOL l2:", single_step_OL["l2"][0])
    # print("\nOL linfty_by_var_list:", single_step_OL["linfty_by_var_list"][0])
    # print("\nOL cos_sim_list:", single_step_OL["cos_sim_list"][0])
    # print("\nOL err:", single_step_OL["err_list"][0])

    # print(acts_new.shape)

    # tosave = {
    #     "nominal_rollouts": noms,
    #     "random_rollouts": acts_new
    # }

    # with open(PICKLE_JAR + "gemma2-2b-acts.pkl", "wb") as f:
    #     pickle.dump(tosave, f)


    # delta_x0 = acts_new[0] - X_nom[0]

    CL_mats = (dat["A_nom"] - dat["K"]).to(device)

    P_cpu = solve_backwards_lyapunov(CL_mats)


    # err_sum = th.zeros(acts_new.shape[1]-1)
    err_max = th.zeros(acts_new.shape[1]-1)

    # err_max = th.zeros(acts_new.shape[1])


    for i in range(acts_new.shape[0]):
        errors = th.tensor(lin_err_CL(X_nom_cpu, dat["A_nom"], acts_new[i], U, variances, diffs, cov_mat, P_cpu)["Pnorm_list"])
        for j, e in enumerate(errors):
            if e > err_max[j]:
                err_max[j] = e
        # err_sum += errors

    # errors = (err_sum / acts_new.shape[0]).tolist() 
    errors = err_max.tolist()
    



    P = P_cpu.to(device)
    # all_variances = [th.diag(th.cov(acts_new[:,i,:].T)).to(device) for i in range(acts_new.shape[1])]
    # Pvar = [th.sqrt(v.T @ P[i] @ v).item() for i, v in enumerate(all_variances)]
    Pvar = [th.sqrt(th.trace(P[i] @ th.cov(acts_new[:,i,:].T).to(device))).item() for i in range(acts_new.shape[1])]

    all_bounds = []
    all_errs = []
    X_nom = dat["X_nom"].to(device)

    for i in range(len(noms)):
        dat=noms[i]
        X_cl = dat["X_cl"].to(device)

        running_product = 1

        err = (X_cl[0] - X_nom[0])
        bounds = [th.sqrt(err.T @ P[0] @ err).item()/Pvar[0]]
        true_errs = [bounds[0]]
        # for i in range(CL_mats.shape[0]):
        for i in range(1,CL_mats.shape[0]+1):
            print(f"layer {i}")
            # evals = th.linalg.eigvals(CL_mats[i])

            # print(f"is contracting: {is_contracting(CL_mats[i], P[i], P[i+1])}")

            # # AtA = CL_mats[i].H @ CL_mats[i]

            # # eAtA = th.linalg.eigvals(AtA).real
            # # print(eAtA)
            # # print(f"manual 2 norm: {th.sqrt(th.max(eAtA)).item()}")

            # print(f"pytorch 2 norm: {th.linalg.norm(CL_mats[i],ord=2, dim=(-2,-1))}")
            print(f"op norm: {op_norm(CL_mats[i-1], P[i-1], P[i])}")

            # bound = bound_at_k(i, (acts_new[0][0]-X_nom[0]).to(device), errors, CL_mats.to(device), P.to(device))
            bound = bound_at_k(i, (X_cl[0]-X_nom[0]), errors, CL_mats, P)
            print(f"bound: {bound}")

            err = (X_cl[i] - X_nom[i])
            errP = th.sqrt(err.T @ P[i] @ err).item()

            print(f"true error: {errP}")

            bounds.append(bound.cpu() / Pvar[i])
            true_errs.append(errP / Pvar[i])
            print(f"normalization factor: {Pvar[i]}")
        all_bounds.append(bounds)
        all_errs.append(true_errs)


    # normalization_factors = 

    bounds = np.array(all_bounds)
    errs = np.array(all_errs)


    np.savez(
        "gemma_lin_data.npz",
        all_bounds=bounds,
        all_errs=errs,
    )

    # data = np.load("gemma_lin_data.npz")

    # bounds = data["all_bounds"]
    # errs = data["all_errs"]

    print("bounds shape", bounds.shape)
    print("errs shape", errs.shape)

    # Time axis
    x = np.arange(bounds.shape[1])

    # --- Select the bound with the largest initial value ---
    # initial value = value at t = 0
    idx = np.argmax(bounds[:, 0])
    worst_bound = bounds[idx]

    # --- Compute min/max envelope of errors ---
    err_min = errs.min(axis=0)
    err_max = errs.max(axis=0)

    # --- Plot ---
    plt.figure()

    # Error envelope
    plt.fill_between(
        x,
        err_min,
        err_max,
        color="red",
        alpha=0.3,
        label="Tracking Error"
    )

    # Optional: outline the envelope
    plt.plot(x, err_min, color="red", linewidth=1, label="lower bound")
    plt.plot(x, err_max, color="blue", linewidth=1, label="upper bound")

    # Worst-case bound
    plt.plot(
        x,
        worst_bound,
        color="black",
        linewidth=2,
        label="Worst-case bound"
    )

    plt.xlabel("Model Layer")
    plt.ylabel("Normalized Error (Lyapunov Norm)")
    plt.title("Worst-Case Bound vs Error Envelope")

    plt.savefig("gemma2-2b-err-bound_range.png")
    plt.show()

    # plt.plot(bounds, label="Approximate Error Bound")
    # plt.plot(true_errs, label="Tracking Error")

    # plt.xlabel("Model Layer")
    # plt.ylabel("Normalized Error (Lyapunov Norm)")

    # plt.legend()
    

        # print(evals)
        # m = th.max(th.abs(evals)).item()
        # print(m)

        # print(th.topk(th.abs(evals), 10))
        # print(th.mean(th.abs(evals)))

        # running_product = running_product * m
        # print(f"running product: {running_product}")

#     # acts,jacs = dataguy.collect_acts_and_jacs(prompts=prompts, num_samples=10, filename="gemma-2-2b-actsnjacs")
#     # print(acts_new[:,0,:])
#     print(f"new acts shape: {acts_new.shape}")


#     diffs = th.zeros_like(acts_new[0])
#     for i in range(acts_new.shape[1]):
#         for j in range(acts_new.shape[2]):
#             min = th.min(acts_new[:,i,j])
#             max = th.max(acts_new[:,i,j])
#             diffs[i,j] = max-min

#     print(f"diffs: {diffs}")


    
#     # print(th.mean(variances,dim=0))
#     print(f"variances: {variances}")



    

# # "l1_by_var": l1norm_byvar_list,
# #         "avg_l1_by_var": avg_l1norm_byvar_list,
# #         "avg_l1_by_diffs": avg_l1norm_bydiffs_list,
# #         "l2_by_var": l2norm_byvar_list,
# #         "l2": l2norm_list,
# #         "linfty_by_var_list": linfty_by_var_list,
# #         "cos_sim_list": cos_sim_list,

#     cumulative_out = lin_err_CL(steer.X[0][:,-1,:].detach().cpu(), steer.A.detach().cpu(), steer.X_cl[0][:,-1,:].detach().cpu(), steer.U.detach().cpu(), variances, diffs, cov_mat)

#     print("\n\nCL by l1:", cumulative_out["l1"][0])
#     print("\n\nCL by avg_l1_by_var:", cumulative_out["avg_l1_by_var"][0])
#     print("\nCL by avg_l1_by_diffs:", cumulative_out["avg_l1_by_diffs"][0])
#     print("\nCL by l1_by_var:", cumulative_out["l1_by_var"][0])
#     print("\nCL by l2_by_var:", cumulative_out["l2_by_var"][0])
#     print("\nCL by l2:", cumulative_out["l2"][0])
#     print("\nCL by linfty_by_var_list:", cumulative_out["linfty_by_var_list"][0])
#     print("\nCL by cos_sim_list:", cumulative_out["cos_sim_list"][0])
#     print("\nCL err:", cumulative_out["err_list"][0])

    # "": l1norm_byvar_list,
    #     "avg_l1_by_var": avg_l1norm_byvar_list,
    #     "avg_l1_by_diffs": avg_l1norm_bydiffs_list,
    #     "ratio": ratio_list,
    #     "l2_by_var": l2norm_byvar_list,
    #     "l2": l2norm_list,
    #     "linfty_by_var_list": linfty_by_var_list,
    #     "cos_sim_list": cos_sim_list,
    # print(cumulative_out["ratio"])
    # print("YAHOO")
    # print(acts_new)


if __name__ == "__main__":
    main()
