import pandas as pd

# Load the pipe-separated text file
txt_path = "Fermi_GBM_catalog_15_8_22.txt"
csv_path = "Fermi_GBM_catalog_15_8_22.csv"

print(f"Reading {txt_path} ...")
df = pd.read_csv(txt_path, sep="|", skipinitialspace=True)

# Because each line starts and ends with '|', pandas creates an empty first and last column
# We will drop them if they are unnamed
if df.columns[0].startswith("Unnamed"):
    df = df.drop(df.columns[0], axis=1)
if isinstance(df.columns[-1], str) and df.columns[-1].startswith("Unnamed"):
    df = df.drop(df.columns[-1], axis=1)

# Clean up column names (strip leading/trailing whitespaces)
df.columns = df.columns.str.strip()

# Now save it as a true comma-separated CSV file
print(f"Saving to {csv_path} ...")
df.to_csv(csv_path, index=False)
print("Done!")
