import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from experiment import MODEL_CONFIGS

def create_model_dataframe():
    """Convert MODEL_CONFIGS to a pandas DataFrame with model families."""
    data = []
    for model_name, config in MODEL_CONFIGS.items():
        # Extract model family from name
        if "gemma" in model_name.lower():
            family = "Gemma"
        elif "mixtral" in model_name.lower():
            family = "Mixtral"
        elif "phi" in model_name.lower():
            family = "Phi"
        elif "qwen" in model_name.lower():
            family = "Qwen"
        elif "yi" in model_name.lower():
            family = "Yi"
        elif "bloom" in model_name.lower():
            family = "BLOOM"
        elif "santa" in model_name.lower():
            family = "Santacoder"
        elif "t5" in model_name.lower():
            family = "T5"
        elif "pythia" in model_name.lower():
            family = "Pythia"
        elif "mgpt" in model_name.lower():
            family = "mGPT"
        else:
            family = "Other"
            
        # Get model size from name (rough approximation)
        size = None
        for s in ["70m", "560m", "1b", "2b", "3b", "4b", "6b", "7b", "14b", "27b", "32b", "34b", "72b"]:
            if s in model_name.lower():
                size = float(s.replace('b', '000m').replace('m', ''))
                break
                
        data.append({
            'model_name': model_name,
            'family': family,
            'batch_size': config['batch_size'],
            'd_model': config['d_model'],
            'dtype': config['dtype'],
            'size_millions': size
        })
    
    return pd.DataFrame(data)

def plot_model_configurations():
    """Create visualizations of model configurations."""
    df = create_model_dataframe()
    
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Batch Size vs Model Size
    sns.scatterplot(data=df, x='size_millions', y='batch_size', 
                   hue='family', style='dtype', s=100, ax=ax1)
    ax1.set_title('Batch Size vs Model Size')
    ax1.set_xlabel('Model Size (M parameters)')
    ax1.set_ylabel('Batch Size')
    ax1.set_xscale('log')
    
    # 2. Model Dimension Distribution by Family
    sns.boxplot(data=df, x='family', y='d_model', ax=ax2)
    ax2.set_title('Model Dimension Distribution by Family')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Data Type Distribution
    dtype_counts = df['dtype'].value_counts()
    dtype_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax3)
    ax3.set_title('Distribution of Data Types')
    
    # 4. Model Dimension vs Batch Size
    sns.scatterplot(data=df, x='d_model', y='batch_size',
                   hue='family', style='dtype', s=100, ax=ax4)
    ax4.set_title('Model Dimension vs Batch Size')
    
    plt.tight_layout()
    plt.savefig('model_configurations.png')
    plt.close()

def print_model_statistics():
    """Print summary statistics about the model configurations."""
    df = create_model_dataframe()
    
    print("\nModel Configuration Statistics:")
    print("-" * 30)
    print(f"Total number of models: {len(df)}")
    print(f"\nModels per family:")
    print(df['family'].value_counts())
    print(f"\nData types used:")
    print(df['dtype'].value_counts())
    print(f"\nBatch size statistics:")
    print(df['batch_size'].describe())
    print(f"\nModel dimension statistics:")
    print(df['d_model'].describe())

if __name__ == "__main__":
    plot_model_configurations()
    print_model_statistics()
