import json
import pandas as pd

# Load the JSON file
with open("plangen_gaia_tree_results.json", "r") as f:
    data = json.load(f)

# Flatten the JSON structure
df = pd.json_normalize(data)

# Save to CSV
df.to_csv("plangen_gaia_tree_results.csv", index=False)

print("CSV file saved as 'plangen_gaia_tree_results.csv'")
