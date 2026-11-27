from datasets import load_dataset

# Load the dataset
dataset = load_dataset(
    "manu/project_gutenberg",
    split="en"
)

# Save to CSV
dataset.to_csv("project_gutenberg.csv", index=False)