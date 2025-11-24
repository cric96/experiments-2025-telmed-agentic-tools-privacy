import pandas as pd
import json
import sys
import os

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. CWD: {os.getcwd()}")
        sys.exit(1)
    
    # Flatten the data slightly to get is_attacked at the top level
    flattened_data = []
    for entry in data:
        # Handle cases where judge_result might be missing or None
        judge_result = entry.get('judge_result')
        is_attacked = False
        if judge_result:
             is_attacked = judge_result.get('is_attacked', False)

        row = {
            'model': entry.get('model', 'Unknown'),
            'runner_case': entry.get('runner_case', 'Unknown'),
            'attack_type': entry.get('attack_type', 'Unknown'),
            'is_attacked': is_attacked
        }
        flattened_data.append(row)
    
    return pd.DataFrame(flattened_data)

def analyze_dimension(df, dimension, name):
    print(f"\n{'='*20} Performance by {name} {'='*20}")
    # Calculate Attack Success Rate (ASR)
    # ASR = mean(is_attacked) * 100
    stats = df.groupby(dimension)['is_attacked'].agg(['count', 'sum', 'mean'])
    stats['ASR (%)'] = stats['mean'] * 100
    stats = stats.rename(columns={'count': 'Total Runs', 'sum': 'Successful Attacks'})
    stats = stats.sort_values('ASR (%)', ascending=False)
    
    # Format for display
    print(stats[['Total Runs', 'Successful Attacks', 'ASR (%)']].to_string(float_format="{:.2f}".format))

def main():
    # Assuming running from root, file is in root
    filepath = 'evaluation_results_judged.json'
    df = load_data(filepath)

    print(f"Total records analyzed: {len(df)}")
    print("Metric: Attack Success Rate (ASR) - Lower is better for defense, Higher is better for attack.")

    # 1. Performance by Model
    analyze_dimension(df, 'model', 'Model')

    # 2. Performance by Attack Type
    analyze_dimension(df, 'attack_type', 'Attack Type')

    # 3. Performance by Runner Case
    analyze_dimension(df, 'runner_case', 'Runner Case')

if __name__ == "__main__":
    main()
