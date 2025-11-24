import json
import seaborn
# Leggi il file JSON
with open('evaluation_results.json', 'r') as f:
    data = json.load(f)

# Conta le entry
num_entries = len(data)

print(f"Numero totale di entry: {num_entries}")
