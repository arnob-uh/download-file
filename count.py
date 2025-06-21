import pandas as pd

# Load CSV file
file_path = "dc-vae_anomaly_predictions.csv"  # Update with the actual file path
df = pd.read_csv(file_path, parse_dates=["time"])  # Ensure 'time' is read as a datetime column

# Convert boolean-like strings to actual boolean values
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.astype(str).str.strip().str.lower() == 'true')

# Compute counts and percentages
summary = {}
for col in df.columns[1:]:  # Skip the 'time' column
    true_count = df[col].sum()
    false_count = len(df) - true_count
    true_percentage = (true_count / len(df)) * 100

    summary[col] = {
        "True Count": true_count,
        "False Count": false_count,
        "True Percentage": round(true_percentage, 2)
    }

# Convert summary to DataFrame
summary_df = pd.DataFrame(summary).T

# Display or save results
print(summary_df)

# Optionally, save to a CSV file
summary_df.to_csv("true_false_summary.csv")