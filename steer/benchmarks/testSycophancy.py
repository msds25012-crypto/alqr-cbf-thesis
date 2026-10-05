"""Original A-LQR on the sycophancy dataset (philpapers2020, two-option questions)."""
import argparse
import json
import os
import pickle
import random
import urllib.request

DATA_URL = ("https://raw.githubusercontent.com/anthropics/evals/main/"
            "sycophancy/sycophancy_on_philpapers2020.jsonl")


def get_topic(q):
    return q.split("following topic:")[-1].split("\n")[0].strip()


def load_rows(path="philpapers2020.jsonl"):
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    rows = []
    for line in open(path):
        r = json.loads(line)
        non = r["answer_not_matching_behavior"]
        non = non if isinstance(non, list) else [non]
        if len(non) == 1 and r["answer_matching_behavior"].strip() in ("(A)", "(B)"):
            rows.append(dict(q=r["question"], syc=r["answer_matching_behavior"],
                             topic=get_topic(r["question"])))
    return rows


def balance(rs, n):
    A = [r for r in rs if r["syc"].strip() == "(A)"]
    B = [r for r in rs if r["syc"].strip() == "(B)"]
    k = min(n, len(A), len(B))
    out = random.sample(A, k) + random.sample(B, k)
    random.shuffle(out)
    return out


def build_test_set(seed=42, n_train=500, n_test=125):
    # same random calls as your notebook, so you get the same 250 test questions
    random.seed(seed)
    rows = load_rows()
    topics = sorted(set(r["topic"] for r in rows))
    random.shuffle(topics)
    train_topics = set(topics[:int(0.8 * len(topics))])
    balance([r for r in rows if r["topic"] in train_topics], n_train)
    return balance([r for r in rows if r["topic"] not in train_topics], n_test)


def load_pickle(name, dirs):
    for d in dirs:
        p = os.path.join(str(d), name + ".pkl")
        if os.path.exists(p):
            print("loading", p)
            return pickle.load(open(p, "rb"))
    raise FileNotFoundError(name + ".pkl not found in " + str(dirs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--key", default="Llama1BInstruct")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 2.5, 3.5, 5, 7, 10])
    ap.add_argument("--q", type=float, default=10)
    ap.add_argument("--r", type=float, default=10)
    ap.add_argument("--qf", type=float, default=1)
    ap.add_argument("--pickle-dir", default="/content/drive/MyDrive/THESIS/alqr_pickles")
    args = ap.parse_args()

    import torch as th
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from steer.data_handling import PICKLE_JAR
    from steer.steering import LQRSteering

    test = build_test_set()
    print("test questions:", len(test))

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=th.float32).to("cuda").eval()
    idA = tok(" (A", add_special_tokens=False)["input_ids"][-1]
    idB = tok(" (B", add_special_tokens=False)["input_ids"][-1]

    def is_syc(r, t):
        return t == (idA if r["syc"].strip() == "(A)" else idB)

    # no steering
    syc = valid = 0
    with th.no_grad():
        for i in range(0, len(test), 16):
            b = test[i:i + 16]
            inp = tok([r["q"] + " (" for r in b], return_tensors="pt", padding=True).to("cuda")
            for r, t in zip(b, model(**inp).logits[:, -1, :].argmax(-1).tolist()):
                if t in (idA, idB):
                    valid += 1
                    syc += is_syc(r, t)
    print(f"NO STEERING   pick {syc / len(test):.3f}  valid {valid / len(test):.3f}")

    # original A-LQR (CBF off, no_overshoot off)
    dirs = [PICKLE_JAR, args.pickle_dir]
    X_contr = load_pickle(args.key + "-nonsyc", dirs)["X"] - load_pickle(args.key + "-syc", dirs)["X"]
    A = load_pickle(args.key + "-nonsyc_jac", dirs)["A"]
    steer = LQRSteering(model, tok, q=args.q, r=args.r, qf=args.qf, A=A, contrastive_vecs=X_contr)
    steer.use_cbf = False
    steer.no_overshoot = False

    for lam in args.lambdas:
        syc = valid = 0
        for i in range(0, len(test), 10):
            b = test[i:i + 10]
            out = steer.track_setpoint([r["q"] + " (" for r in b], 1, lmbda=lam,
                                       do_sample=False, return_tokens=True)
            for r, t in zip(b, out[:, -1].tolist()):
                if t in (idA, idB):
                    valid += 1
                    syc += is_syc(r, t)
        print(f"A-LQR lambda {lam:>4}  pick {syc / len(test):.3f}  valid {valid / len(test):.3f}")


if __name__ == "__main__":
    main()
