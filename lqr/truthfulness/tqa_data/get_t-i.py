import json

# filename = 'gemma-2-2b-BASE-sweep.txt'
filename = 'gemma-2-2b-secondary2.txt'
# filename = 'llama-3-8b-partial.txt'

with open(filename, 'r') as file:
    data = json.load(file) # json.load() directly reads from a file object
unsteered = data[0]
u_t = unsteered["unsteered t"]
u_i = unsteered["unsteered i"]

print(f"base ti: {u_t * u_i}, base t: {u_t}, base i: {u_i}")


m = 0
best_l = 0
best_q = 0
best_r = 0
best_qf = 0


curr_05 = 0
curr_1 = 0
curr_15 = 0
curr_2 = 0
for sweep in data[1]["sweeps"]:
# for sweep in data:
    if sweep["lambda"] == 0.5:
        t = sweep["steered t"]
        i = sweep["steered i"]
        curr_05 = t*i
    if sweep["lambda"] == 1:
        t = sweep["steered t"]
        i = sweep["steered i"]
        curr_1 = t*i
    if sweep["lambda"] == 1.5:
        t = sweep["steered t"]
        i = sweep["steered i"]
        curr_15 = t*i
    if sweep["lambda"] == 2:
        t = sweep["steered t"]
        i = sweep["steered i"]
        curr_2 = t*i


    if sweep["lambda"] == 2.5:
        t = sweep["steered t"]
        i = sweep["steered i"]
        ti = t*i
        # if ti >= curr_2 and curr_2 >= curr_15 and curr_15 >= curr_1 and curr_1 >= curr_05:
        if ti > m:
            m = ti
            best_l = sweep["lambda"]
            best_q = sweep["Q"]
            best_r = sweep["R"]
            best_qf = sweep["Qf"]

print(f"best ti: {m}")
print(f"best l: {best_l}, best q: {best_q}, best r: {best_r}, best qf: {best_qf}")


for sweep in data[1]["sweeps"]:
# for sweep in data:
    t = sweep["steered t"]
    i = sweep["steered i"]

    q = sweep["Q"]
    r = sweep["R"]
    qf = sweep["Qf"]

    la = sweep["lambda"]
    if best_q == q and best_r == r and best_qf == qf:
        print(f"{la} & {(t*i):.4f} & {t:.4f} & {i:.4f} \\\\")