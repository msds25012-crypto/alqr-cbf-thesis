import json
import csv

# Input and output files
model = "qwen3b"
json_path = "toxicity_sweep_setpoint/toxicity_sweeps_setpoint_" + model + ".json"
csv_path = "toxicity_sweep_setpoint/" + model + ".csv"

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Extract sweeps
sweeps = data["sweeps"]

# Get CSV column names from keys of first sweep
fieldnames = sweeps[0].keys()

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sweeps)

print(f"Wrote {len(sweeps)} rows to {csv_path}")
