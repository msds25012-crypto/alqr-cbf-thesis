import json

filename = 'path.txt'

with open(filename, 'r') as file:
    data = json.load(file) # json.load() directly reads from a file object
unsteered = data[0]
u_t = unsteered["unsteered t"]
u_i = unsteered["unsteered i"]

print(f"base ti: {u_t * u_i}")


m = 0
best_l = 0
best_q = 0
best_r = 0
best_qf = 0


for sweep in data[1]["sweeps"]:
    t = sweep["steered t"]
    i = sweep["steered i"]
    ti = t*i
    if ti > m:
        m = ti
        best_l = sweep["lambda"]
        best_q = sweep["Q"]
        best_r = sweep["R"]
        best_qf = sweep["Qf"]

print(f"best ti: {m}")
print(f"best l: {best_l}, best q: {best_q}, best r: {best_r}, best qf: {best_qf}")
