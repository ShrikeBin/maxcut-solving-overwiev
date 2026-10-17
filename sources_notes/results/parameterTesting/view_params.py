import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

# Check if a filename was provided as a command-line argument
if len(sys.argv) < 2:
  print("Error: Please provide the path to your data file as an argument.")
  print("Usage: python script.py <path_to_data_file>")
  sys.exit(1)

file_path = sys.argv[1]

# 1. Load the data
try:
  df = pd.read_csv(file_path)
except Exception as e:
  print(f"Error reading file '{file_path}': {e}")
  sys.exit(1)

# Clean column names just in case of whitespace
df.columns = df.columns.str.strip()

print("=" * 50)
print(f"DATA OVERVIEW: {file_path}")
print("=" * 50)
print(f"Total rows loaded: {len(df)}")
print(f"Unique methods: {list(df['method'].unique())}")
print(f"Unique graphs: {list(df['graph'].unique())}")

print("\n" + "=" * 50)
print("OVERALL MEDIANS")
print("=" * 50)
print(df[["weight", "cut_edges", "time_s"]].median())

print("\n" + "=" * 50)
print("TOP 10 CONFIGURATIONS (Highest Weight / Cut Edges)")
print("=" * 50)
top_10 = df.sort_values(by=["weight", "cut_edges"], ascending=False).head(10)
print(
    top_10[
        ["method", "graph", "config_idx", "config", "weight", "cut_edges", "time_s"]
    ].to_string(index=False)
)

print("\n" + "=" * 50)
print("TOP 5 CONFIGURATIONS PER ALGORITHM (Highest Weight / Cut Edges)")
print("=" * 50)
for method_name, group in df.groupby("method"):
  top_5_algo = group.sort_values(
      by=["weight", "cut_edges"], ascending=False
  ).head(5)
  print(f"\n--- Method: {method_name} ---")
  print(
      top_5_algo[
          ["config_idx", "config", "weight", "cut_edges", "time_s"]
      ].to_string(index=False)
  )

print("\n" + "=" * 50)
print("SUMMARY STATISTICS BY METHOD")
print("=" * 50)
method_summary = (
    df.groupby("method")[["weight", "cut_edges", "time_s"]]
    .agg(["median", "mean", "min", "max"])
    .round(4)
)
print(method_summary)

# 2. Visualization: Highly Readable Weight Distribution Boxplot by Method (using Seaborn)
if "method" in df.columns:
  plt.figure(figsize=(10, 6))
  sns.boxplot(
      x="method",
      y="weight",
      data=df,
      palette="Set2",
      boxprops=dict(alpha=0.8),
      hue="method",
      legend=False,
  )
  sns.stripplot(
      x="method",
      y="weight",
      data=df,
      color="black",
      alpha=0.3,
      jitter=0.2,
      size=4,
  )
  plt.title("Weight Distribution by Method", fontsize=14, pad=15)
  plt.xlabel("Method", fontsize=12)
  plt.ylabel("Weight", fontsize=12)
  plt.xticks(rotation=30, ha="right")
  plt.grid(True, linestyle="--", alpha=0.3, axis="y")
  plt.tight_layout()
  plt.show()