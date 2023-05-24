import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to the training data file
training_data_path = "/Users/andrejhric_1/git_projects/DS_CW/DS/training_input.tsv"
clean_data_path = "/Users/andrejhric_1/git_projects/DS_CW/DS/clean_input.tsv"

# Read the TSV file into a pandas DataFrame
df = pd.read_csv(training_data_path, sep="\t")

# Extract the subcellular location column without the "[CC]" suffix
df["Subcellular location"] = df["Subcellular location [CC]"].str.replace(r'\s*\[CC\]', '')

# Filter out sequences with unknown or no information
df_filtered = df.dropna(subset=["Subcellular location"])

# Save the filtered data as a new TSV file
df_filtered.to_csv(clean_data_path, sep="\t", index=False)

# Group the filtered data based on subcellular location
group_counts = df_filtered["Subcellular location"].value_counts()

# Output group counts as a table
group_counts_table = pd.DataFrame(group_counts)
group_counts_table.columns = ["Count"]
print(group_counts_table)

# Plot group counts as a bar plot
plt.figure(figsize=(10, 6))
group_counts.plot(kind="bar")
plt.xlabel("Subcellular Location")
plt.ylabel("Count")
plt.title("Distribution of Subcellular Locations")
plt.xticks(rotation=90)
plt.show()
