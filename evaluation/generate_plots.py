import json
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Fix QFont warnings by using non-interactive backend
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style for gold standard visualization
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'serif' # LaTeX-like font
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

OUTPUT_DIR = "evaluation/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_process_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    records = []
    for entry in data:
        judge = entry.get('judge_result', {})
        # If judge_result is None, treat as not attacked or skip? 
        # Assuming False if missing.
        is_attacked = judge.get('is_attacked', False) if judge else False
        
        records.append({
            'runner_case': entry.get('runner_case'),
            'model': entry.get('model'),
            'attack_type': entry.get('attack_type'),
            'run_number': entry.get('run_number'),
            'patient_context': entry.get('patient_context'),
            'attack_question': entry.get('attack_question'),
            'is_attacked': is_attacked
        })
    
    df = pd.DataFrame(records)
    
    # Filter for Baseline and Strong Privacy Protection
    # We exclude "Moderate Privacy Guidelines" based on "just consinder storng baseline" 
    # interpreting "storng baseline" as "Strong Privacy Protection" and "Baseline - No Privacy Guidelines"
    target_cases = ['Baseline - No Privacy Guidelines', 'Strong Privacy Protection']
    df = df[df['runner_case'].isin(target_cases)]
    
    # Rename for better plotting labels
    df['runner_case'] = df['runner_case'].replace({
        'Baseline - No Privacy Guidelines': 'Baseline',
        'Strong Privacy Protection': 'Strong Privacy'
    })

    # Compact Attack Types
    # Mapping:
    # Direct Prompt Injection -> P. I.
    # Explicit Private Info Request -> E. R.
    # Tricky Disguised Query -> T. Q.
    # Contextual Cross-Reference Request -> C. R.
    # Conditional Sensitive Data Mining -> C. M.
    df['attack_type'] = df['attack_type'].replace({
        'Direct Prompt Injection': 'P. I.',
        'Explicit Private Info Request': 'E. R.',
        'Tricky Disguised Query': 'T. Q.',
        'Contextual Cross-Reference Request': 'C. R.',
        'Conditional Sensitive Data Mining': 'C. M.'
    })
    
    return df

def calculate_metrics(df):
    # ASR is simply the mean of is_attacked
    # Attack@1: Group by (Model, Runner Case, Attack Type, Patient, Question)
    # If any run in the group is attacked, then Attack@1 = 1, else 0
    
    group_cols = ['model', 'runner_case', 'attack_type', 'patient_context', 'attack_question']
    
    # Attack@1 DataFrame
    # We aggregate by taking the max of is_attacked (True if any is True)
    attack_at_1_df = df.groupby(group_cols)['is_attacked'].max().reset_index()
    attack_at_1_df.rename(columns={'is_attacked': 'attack_at_1'}, inplace=True)
    
    return df, attack_at_1_df

import numpy as np

# Define custom palettes
# Privacy Levels: Baseline (High Risk) -> Red, Strong Privacy (Low Risk) -> Blue
PRIVACY_PALETTE = {
    'Baseline': '#d62728',       # Muted Red
    'Strong Privacy': '#1f77b4'  # Muted Blue
}

# Consistent colors for models (Tab10 excluding Red and Blue)
# Green, Orange, Purple, Brown, Pink, Gray, Olive, Cyan
MODEL_COLORS = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MODEL_PALETTE = {} # Will be populated in main

def plot_bar(df, x, y, hue, title, filename, ylabel="Success Rate"):
    plt.figure(figsize=(10, 6))
    
    # Determine palette based on hue
    if hue == 'runner_case':
        palette = PRIVACY_PALETTE
    elif hue == 'model':
        palette = MODEL_PALETTE
    else:
        palette = "viridis"

    # Create barplot
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, errorbar=('ci', 95), capsize=.1, edgecolor=".2")
    
    # Aesthetics
    sns.despine(left=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)
    
    plt.title(title, pad=20, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.xlabel(x.replace('_', ' ').title(), fontweight='bold')
    
    # Move legend
    plt.legend(title=hue.replace('_', ' ').title(), bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    # Rotate x labels if needed
    if df[x].nunique() > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Removed value labels inside bars as requested
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.2f', label_type='center', color='white', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_heatmap(df, x, y, value, title, filename):
    # Pivot data
    pivot_table = df.groupby([y, x])[value].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    # Use 'Reds' for ASR (0=Good/White, 1=Bad/Red) or 'vlag' for diverging
    # Since 0 is safe and 1 is unsafe, a sequential palette like 'Reds' or 'rocket_r' is good.
    # 'rocket_r' goes from light to dark.
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Reds", 
                cbar_kws={'label': 'Success Rate'}, linewidths=.5, linecolor='gray')
    
    plt.title(title, pad=20, fontweight='bold')
    plt.ylabel(y.replace('_', ' ').title(), fontweight='bold')
    plt.xlabel(x.replace('_', ' ').title(), fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_radar(df, category_col, value_col, group_col, title, filename):
    categories = sorted(df[category_col].unique())
    groups = sorted(df[group_col].unique())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Offset the starting angle to be at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Manual label placement to ensure spacing
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    
    for angle, label in zip(angles[:-1], categories):
        # Dynamic alignment based on angle to prevent overlap
        if angle == 0:
            ha, va = 'center', 'bottom'
        elif 0 < angle < np.pi:
            ha, va = 'left', 'center'
        elif angle == np.pi:
            ha, va = 'center', 'top'
        else:
            ha, va = 'right', 'center'
        
        # Place label slightly outside the outer circle (1.05)
        ax.text(angle, 1.15, label, ha=ha, va=va, color='black', size=20)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=12, fontweight='bold')
    plt.ylim(0, 1.05)
    
    # Determine colors
    if group_col == 'runner_case':
        # Use the dictionary palette directly
        colors = [PRIVACY_PALETTE[g] for g in groups]
    elif group_col == 'model':
        # Use the MODEL_PALETTE dictionary
        colors = [MODEL_PALETTE[g] for g in groups]
    else:
        # Fallback
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(groups))]

    for idx, group in enumerate(groups):
        subset = df[df[group_col] == group]
        values = []
        for cat in categories:
            # Handle missing categories by filling with 0 or NaN (though mean() handles empty)
            val = subset[subset[category_col] == cat][value_col].mean()
            if pd.isna(val): val = 0
            values.append(val)
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=group, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.25)
    
    plt.title(title, y=1.1, fontweight='bold', fontsize=24)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=False, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def main():
    filepath = 'evaluation_results_judged.json'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print("Loading data...")
    df = load_and_process_data(filepath)
    
    # Populate MODEL_PALETTE
    unique_models = sorted(df['model'].unique())
    global MODEL_PALETTE
    MODEL_PALETTE = {model: MODEL_COLORS[i % len(MODEL_COLORS)] for i, model in enumerate(unique_models)}
    
    print("Calculating metrics...")
    df_asr, df_at1 = calculate_metrics(df)
    
    # --- PLOTS ---
    
    # 1. Performance by Model (ASR) - Hue: Runner Case
    print("Generating Model plots...")
    plot_bar(df_asr, x='model', y='is_attacked', hue='runner_case', 
             title='Attack Success Rate (ASR) by Model', 
             filename='asr_by_model.pdf', ylabel="ASR")
    
    # 2. Performance by Model (Attack@1) - Hue: Runner Case
    plot_bar(df_at1, x='model', y='attack_at_1', hue='runner_case', 
             title='Attack@1 Probability by Model', 
             filename='attack_at_1_by_model.pdf', ylabel="Attack@1")

    # 3. Performance by Attack Type (ASR) - Hue: Runner Case
    print("Generating Attack plots...")
    plot_bar(df_asr, x='attack_type', y='is_attacked', hue='runner_case', 
             title='ASR by Attack Type', 
             filename='asr_by_attack_type.pdf', ylabel="ASR")
    
    # 4. Performance by Attack Type (Attack@1) - Hue: Runner Case
    plot_bar(df_at1, x='attack_type', y='attack_at_1', hue='runner_case', 
             title='Attack@1 by Attack Type', 
             filename='attack_at_1_by_attack_type.pdf', ylabel="Attack@1")

    # 5. Performance by Runner Case (ASR) - Hue: Model
    print("Generating Runner Case plots...")
    plot_bar(df_asr, x='runner_case', y='is_attacked', hue='model', 
             title='ASR by Privacy Level', 
             filename='asr_by_runner_case.pdf', ylabel="ASR")

    # --- NEW VISUALIZATIONS ---
    print("Generating Heatmaps...")
    # Heatmap: Attack Type vs Runner Case (ASR)
    plot_heatmap(df_asr, x='attack_type', y='runner_case', value='is_attacked',
                 title='ASR Heatmap: Attack Type vs Privacy Level',
                 filename='heatmap_asr_attack_runner.pdf')
    
    # Heatmap: Attack Type vs Model (ASR)
    plot_heatmap(df_asr, x='attack_type', y='model', value='is_attacked',
                 title='ASR Heatmap: Attack Type vs Model',
                 filename='heatmap_asr_attack_model.pdf')

    print("Generating Radar Charts...")
    # Radar: Compare Runner Cases across Attack Types (Attack@1)
    plot_radar(df_at1, category_col='attack_type', value_col='attack_at_1', group_col='runner_case',
               title='Attack@1 Radar: Privacy Levels across Attack Types',
               filename='radar_at1_runner_attack.pdf')
    
    # Radar: Compare Models across Attack Types (Attack@1)
    # Filter for Strong Privacy only as requested
    df_at1_strong = df_at1[df_at1['runner_case'] == 'Strong Privacy']
    plot_radar(df_at1_strong, category_col='attack_type', value_col='attack_at_1', group_col='model',
               title='Attack@1 Radar: Models (Strong Privacy)',
               filename='radar_at1_model_attack.pdf')

    # Radar: Per Model (Axes=Attack Types, Lines=Runner Cases)
    # This shows how privacy guidelines affect each model across different attacks
    for model in df_at1['model'].unique():
        model_df = df_at1[df_at1['model'] == model]
        safe_model_name = model.replace(':', '_').replace('.', '_')
        plot_radar(model_df, category_col='attack_type', value_col='attack_at_1', group_col='runner_case',
                   title=f'Attack@1 Radar: {model}',
                   filename=f'radar_at1_runner_attack_{safe_model_name}.pdf')

    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
