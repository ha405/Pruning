import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(json_path="profiling_results.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def print_summary(df):
    print("\n📊 === PROFILING SUMMARY ===")
    print(df[["Model", "Dataset", "Top-1 (%)", "Top-5 (%)", "Latency (ms)", "MACs (M)", "Size (MB)"]])
    print("\n📈 Average Metrics by Dataset:")
    print(df.groupby("Dataset")[["Top-1 (%)", "Latency (ms)", "MACs (M)", "Size (MB)"]].mean().round(3))
    print("\n🔗 Correlation (accuracy vs efficiency):")
    print(df.corr(numeric_only=True)["Top-1 (%)"].sort_values(ascending=False))


import os

def plot_accuracy_vs_size(df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="Size (MB)", y="Top-1 (%)", hue="Dataset", s=100)
    plt.title("Accuracy vs Model Size")
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_size.png"))
    plt.close()


def plot_accuracy_vs_latency(df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="Latency (ms)", y="Top-1 (%)", hue="Dataset", s=100)
    plt.title("Accuracy vs Inference Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_latency.png"))
    plt.close()


def plot_macs_vs_latency(df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="MACs (M)", y="Latency (ms)", hue="Dataset", s=100)
    plt.title("MACs vs Latency")
    plt.xlabel("MACs (Millions)")
    plt.ylabel("Latency (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "macs_vs_latency.png"))
    plt.close()


def main():
    df = load_results("profiling_results.json")
    print_summary(df)

    plot_accuracy_vs_size(df)
    plot_accuracy_vs_latency(df)
    plot_macs_vs_latency(df)

    print("\Plots saved in the 'plots/' directory.")



if __name__ == "__main__":
    main()
