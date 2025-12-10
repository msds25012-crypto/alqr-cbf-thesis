import torch as th
from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
import pickle
from steering import LQRSteering
from datasets import load_dataset
import random
import time

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# use the same tokenizer as TinyLlama
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
# model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
# model_name = "meta-llama/Llama-3.2-1B"
model_name = "google/gemma-2-2b"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "Qwen/Qwen2.5-3B"
# model = LlamaForCausalLM.from_pretrained(
    # model_name).to(device)


# model = AutoModelForCausalLM.from_pretrained(
#     model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,          # or load_in_8bit=True
    bnb_4bit_compute_dtype=th.float16,
    bnb_4bit_quant_type="nf4",  # best for LLMs
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quant_config, dtype=th.float32, device_map="auto")



nontox_filename = "gemma-2-2b_nontox"
# nontox_filename = "qwen2.5-3b_nontox"
# nontox_filename = "llama-3.2-1b_nontox"
# nontox_filename = "llama-3-8b_nontox"
with open("../../scratch/"+nontox_filename+".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)


X = loaded_tensors["X"]
A = loaded_tensors["A"]
print(f"X shape: {X.shape}")
print(f"A shape: {A.shape}")

tox_filename = "gemma-2-2b_tox"
# tox_filename = "qwen2.5-3b_tox"
# tox_filename = "llama-3-8b_tox"
# tox_filename = "llama-3.2-1b_tox"
with open("../../scratch/"+tox_filename+".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

    # Access tensors
X_tox = loaded_tensors["X"]
X_contr = X - X_tox




N_SHOTS = 5
LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# List of all MMLU subjects (configs) in cais/mmlu
# SUBJECTS = [
    # 'abstract_algebra']
SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
    'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
    'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
    'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

def format_example(example):
    answer_letter = LETTER_MAP[example["answer"]]
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    return f"Question: {example['question']}\n{choices}\nAnswer: {answer_letter}\n\n"


def format_query(example):
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    return f"Question: {example['question']}\n{choices}\nAnswer:"


def build_5shot_prompt(dev_set, test_example, n_shots=N_SHOTS):
    exemplars = random.sample(list(dev_set), n_shots)
    prompt = ""
    for ex in exemplars:
        prompt += format_example(ex)
    prompt += format_query(test_example)
    # Store the correct answer letter for the test example
    correct_answer = LETTER_MAP[test_example["answer"]]
    return prompt, correct_answer


N_PROMPTS = 100      # prompts per batch
N_LOOP = 10         # number of batches
BATCH_SIZE = 4      # how many to send to GPU at once
N_SHOTS = 5
do_sample = False
temp = 0.7
k=1
lambda_list = [0.5, 1, 1.5, 2, 2.5]
# lambda_list = [0.5]
steer_contr = LQRSteering(model, tokenizer, q=0.1,r=1,qf=0.1, A=A, contrastive_vecs=X_contr)

def main():
    for l in lambda_list:
        results = []
        output_str = []

        results_steer = []
        output_str_steer = []

        # -------------------------------
        # Pre-load all subjects ONCE
        # -------------------------------
        subject_datasets = {
            sub: load_dataset("cais/mmlu", sub)
            for sub in SUBJECTS
        }

        for _ in range(N_LOOP):

            prompts_with_answers = []
            samples = []

            # -------------------------------
            # Build 10 prompts (CPU only)
            # -------------------------------
            for i in range(N_PROMPTS):
                subject = random.choice(SUBJECTS)
                ds = subject_datasets[subject]
                dev, test = ds["dev"], ds["test"]

                if len(dev) < N_SHOTS or len(test) == 0:
                    continue

                test_example = random.choice(test)
                prompt, correct_answer = build_5shot_prompt(dev, test_example)

                prompts_with_answers.append({
                    "subject": subject,
                    "prompt": prompt,
                    "answer": correct_answer
                })
                samples.append(prompt)

            # -------------------------------
            # GPU-efficient batching
            # -------------------------------
            batch_outputs = []
            batch_outputs_steer = []

            for start in range(0, len(samples), BATCH_SIZE):
                batch = samples[start:start+BATCH_SIZE]

                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                output_un = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=k,
                    do_sample=do_sample,
                    temperature=temp,
                    use_cache=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                decoded = tokenizer.batch_decode(
                    output_un.sequences,
                    skip_special_tokens=True
                )

                batch_outputs.extend(decoded)


                contr_out = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample, temp = temp)
                batch_outputs_steer.extend(contr_out)


                # Important: free GPU memory of this batch
                del inputs
                del output_un
                th.cuda.empty_cache()

            # Store results
            output_str.extend(batch_outputs)
            output_str_steer.extend(batch_outputs_steer)

            # -------------------------------
            # Compare answers
            # -------------------------------
            # print("base model")
            for item, model_output in zip(prompts_with_answers, batch_outputs):
                model_choice = model_output.strip()[-1].upper()
                correct_choice = item["answer"]
                # print(f"model choice: {model_choice}")
                # print(f"correct choice: {correct_choice}")
                results.append(model_choice == correct_choice)

            # print("")
            # print("steered model")
            for item, model_output in zip(prompts_with_answers, batch_outputs_steer):
                model_choice = model_output.strip()[-1].upper()
                correct_choice = item["answer"]
                # print(f"model choice: {model_choice}")
                # print(f"correct choice: {correct_choice}")
                results_steer.append(model_choice == correct_choice)

        # Final accuracy
        print(f"LAMBDA: {l}")
        accuracy = sum(results) / len(results)
        print(f"Final accuracy: {accuracy*100}%")

        accuracy_steer = sum(results_steer) / len(results_steer)
        print(f"Final accuracy (steered): {accuracy_steer*100}%")


if __name__ == "__main__":
    main()
