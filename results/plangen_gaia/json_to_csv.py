import json
import pandas as pd

# Load the JSON file
with open("evaluation_results.json", "r") as f:
    data = json.load(f)

# Flatten the JSON structure
df = pd.json_normalize(data)

# Save to CSV
df.to_csv("evaluation_results_no_algo.csv", index=False)

print("CSV file saved as 'evaluation_results_no_algo.csv'")
