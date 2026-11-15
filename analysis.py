import json
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "similar_posts.json"

# Define match-count bands
bins = [0, 4, 9, 16, 26, 51, float("inf")]
labels = ["1-3", "4-8", "9-15", "16-25", "26-50", "51+"]

# Load JSON
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Categorize by number of matches
df["band"] = pd.cut(
    df["length"],
    bins=bins,
    labels=labels,
    right=False
)

# Count objects in each band
counts = df["band"].value_counts(sort=False)

print("Match count distribution:")
print(counts.to_string())

# Plot
plt.figure(figsize=(10, 5))
bars = plt.bar(counts.index.astype(str), counts.values)

plt.xlabel("Number of matches per post")
plt.ylabel("Number of posts")
plt.title("Distribution of Posts by Match Count")
plt.grid(axis="y", alpha=0.3)

# Add counts above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()