import json

filename = 'gemma-2-9b-angle_sweep.txt'

with open(filename, 'r') as file:
    data = json.load(file) # json.load() directly reads from a file object
unsteered = data[0]
u_r = unsteered["unsteered refused"]
u_n = unsteered["unsteered nonrefused"]
u_asr = unsteered["unsteered ASR"]

print(f"base ref: {u_r / (u_r + u_n)}")
print(f"base asr: {u_asr}")


m = 0
best_l = 0
best_q = 0
best_r = 0
best_qf = 0

best_ref = 0
best_non = 0

for sweep in data[1]["sweeps"]:
    r = sweep["steered refused"]
    n = sweep["steered nonrefused"]
    asr = sweep["Steered ASR"]
    if asr > m:
        m = asr
        best_l = sweep["lambda"]
        best_q = sweep["Q"]
        best_r = sweep["R"]
        best_qf = sweep["Qf"]
        best_ref = r
        best_non = n

print(f"best asr: {m}")
print(f"best l: {best_l}, best q: {best_q}, best r: {best_r}, best qf: {best_qf}")
print(f"ref: {best_ref / (best_non + best_ref)}")
