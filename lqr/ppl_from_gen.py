from evaluate import load
import json

filename = 'tox_data/gemma_2_2b_tox_PID_out.txt'

with open(filename, 'r') as file:
    data = json.load(file)
ppl = load("perplexity", module_type="metric")


# for sweep in data:

unoutput = ppl.compute(predictions=data[0]["unsteered output"], model_id='google/gemma-2-2B')
ppl_unsteered = unoutput['mean_perplexity']
                    # ppl_unsteered = unsteered_results['mean_perplexity']
print(ppl_unsteered)

for sweep in data[1]["sweeps"]:
    unoutput = ppl.compute(predictions=sweep["steered output"], model_id='google/gemma-2-2B')

    ppl_steered = unoutput['mean_perplexity']

    l=sweep["lambda"]
    print(f"l={l}, ppl_steered: {ppl_steered}")