import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define strong models
STRONG_MODELS = [
    "Strong Privacy Protection"
]

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    flattened_data = []
    for entry in data:
        judge_result = entry.get('judge_result')
        is_attacked = False
        if judge_result:
             is_attacked = judge_result.get('is_attacked', False)

        # Create a unique identifier for the scenario (excluding run_number)
        # We use patient_context and attack_question as the unique key for the "attack instance"
        # Using a tuple of strings for hashing
        scenario_key = (
            entry.get('patient_context', ''), 
            entry.get('attack_question', ''), 
            entry.get('attack_type', ''),
            entry.get('model', ''),
            entry.get('runner_case', '')
        )
        scenario_id = hash(scenario_key)

        row = {
            'model': entry.get('model', 'Unknown'),
            'runner_case': entry.get('runner_case', 'Unknown'),
            'attack_type': entry.get('attack_type', 'Unknown'),
            'run_number': entry.get('run_number'),
            'is_attacked': is_attacked,
            'scenario_id': scenario_id
        }
        flattened_data.append(row)
    
    df = pd.DataFrame(flattened_data)
    return df

def filter_strong_models(df):
    return df[df['runner_case'].isin(STRONG_MODELS)]

def calculate_metrics(df):
    # ASR: Average of is_attacked across all runs
    # Attack@1: Group by scenario, if any run is attacked, then it's a success.
    
    # ASR Calculation
    asr_df = df.copy()
    asr_df['is_attacked_int'] = asr_df['is_attacked'].astype(int)
    
    # Attack@1 Calculation
    # Group by Model, Runner Case, Attack Type, Scenario ID
    # Check if max(is_attacked) is 1 (True)
    # We group by the factors we want to aggregate over later, plus the scenario_id
    grouped = df.groupby(['model', 'runner_case', 'attack_type', 'scenario_id'])['is_attacked'].any().reset_index()
    grouped.rename(columns={'is_attacked': 'attack_at_1'}, inplace=True)
    grouped['attack_at_1_int'] = grouped['attack_at_1'].astype(int)
    
    return asr_df, grouped

def plot_performance_by_model(asr_df, attack_at_1_df, output_dir):
    # Plot ASR
    plt.figure(figsize=(10, 6))
    sns.barplot(data=asr_df, x='model', y='is_attacked_int', errorbar=None)
    plt.title('Attack Success Rate (ASR) by Model')
    plt.ylabel('ASR')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'asr_by_model.png'))
    plt.close()

    # Plot Attack@1
    plt.figure(figsize=(10, 6))
    sns.barplot(data=attack_at_1_df, x='model', y='attack_at_1_int', errorbar=None)
    plt.title('Attack@1 Probability by Model')
    plt.ylabel('Attack@1')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'attack_at_1_by_model.png'))
    plt.close()

def plot_performance_by_attack(asr_df, attack_at_1_df, output_dir):
    # ASR by Attack Type and Model
    plt.figure(figsize=(12, 8))
    sns.barplot(data=asr_df, x='attack_type', y='is_attacked_int', hue='model', errorbar=None)
    plt.title('ASR by Attack Type and Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ASR')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'asr_by_attack_model.png'))
    plt.close()

    # ASR by Attack Type and Runner Case
    plt.figure(figsize=(12, 8))
    sns.barplot(data=asr_df, x='attack_type', y='is_attacked_int', hue='runner_case', errorbar=None)
    plt.title('ASR by Attack Type and Runner Case')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ASR')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'asr_by_attack_runner.png'))
    plt.close()
    
    # Attack@1 by Attack Type and Model
    plt.figure(figsize=(12, 8))
    sns.barplot(data=attack_at_1_df, x='attack_type', y='attack_at_1_int', hue='model', errorbar=None)
    plt.title('Attack@1 by Attack Type and Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Attack@1')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_at_1_by_attack_model.png'))
    plt.close()

def plot_performance_by_runner_case(asr_df, attack_at_1_df, output_dir):
    # ASR by Runner Case and Model
    plt.figure(figsize=(12, 8))
    sns.barplot(data=asr_df, x='runner_case', y='is_attacked_int', hue='model', errorbar=None)
    plt.title('ASR by Runner Case and Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ASR')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'asr_by_runner_model.png'))
    plt.close()
    
    # Attack@1 by Runner Case and Model
    plt.figure(figsize=(12, 8))
    sns.barplot(data=attack_at_1_df, x='runner_case', y='attack_at_1_int', hue='model', errorbar=None)
    plt.title('Attack@1 by Runner Case and Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Attack@1')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_at_1_by_runner_model.png'))
    plt.close()

def main():
    input_file = 'evaluation_results_judged.json'
    output_dir = 'evaluation/plots'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Loading data...")
    df = load_data(input_file)
    
    print("Filtering for strong models...")
    df_strong = filter_strong_models(df)
    
    if df_strong.empty:
        print("Warning: No data found for strong models. Checking available models...")
        print(df['model'].unique())
        return

    print(f"Data shape after filtering: {df_strong.shape}")

    print("Calculating metrics...")
    asr_df, attack_at_1_df = calculate_metrics(df_strong)
    
    print("Generating plots...")
    plot_performance_by_model(asr_df, attack_at_1_df, output_dir)
    plot_performance_by_attack(asr_df, attack_at_1_df, output_dir)
    plot_performance_by_runner_case(asr_df, attack_at_1_df, output_dir)
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
