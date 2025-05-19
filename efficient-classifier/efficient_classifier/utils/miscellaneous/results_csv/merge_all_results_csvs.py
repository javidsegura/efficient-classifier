import pandas as pd

PATH_1 = "results/model_evaluation/results.csv"
PATH_2 = "results/model_evaluation/prior_csvs/april_15.csv"

df_1 = pd.read_csv(PATH_1)
df_2 = pd.read_csv(PATH_2)
df_2.drop(columns=["Unnamed: 20", "classification_report"], inplace=True)

print(df_1.head())
print(df_1.columns)
print(df_1.shape)

print(df_2.head())
print(df_2.columns)
print(df_2.shape)

# Concatenate the two dataframes vertically
print("-"*30)
df = pd.concat([df_1, df_2], ignore_index=True)

# Save the new dataframe
print(df.head())
print(df.columns) 
print(df.shape)

df.loc[df['modelName'] == 'Feed Forward NN', 'modelName'] = 'Feed Forward Neural Network'
df.loc[df['modelName'] == 'Gaussian Naive Bayes', 'modelName'] = 'Naive Bayes'
df.loc[df['modelName'] == 'Logistic Regression', 'modelName'] = 'Logistic Regression (baseline)'
df.loc[df['modelName'] == 'SVM', 'modelName'] = 'Linear SVM'


print(df["modelName"].value_counts())

print(df["f1-score_val"].value_counts())

# Drop records with -1 f1-score_val
df = df[df["f1-score_val"] != -1]

df.sort_values(by="timeStamp", inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Time to Fit vs Timestamp
sns.scatterplot(data=df, x='timeStamp', y='timeToFit', hue='modelName', ax=ax1)
ax1.set_title('Time to Fit vs Timestamp')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Time to Fit (seconds)')

# Set max 10 x-ticks for ax1
xticks1 = df['timeStamp'].drop_duplicates()
xticks1 = xticks1.iloc[np.linspace(0, len(xticks1)-1, min(10, len(xticks1))).astype(int)]
ax1.set_xticks(xticks1)
ax1.set_xticklabels(xticks1, rotation=45)

# Plot 2: F1 Score vs Timestamp
sns.scatterplot(data=df, x='timeStamp', y='f1-score_val', hue='modelName', ax=ax2)
ax2.set_title('F1 Score vs Timestamp')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('F1 Score')
ax2.set_ylim(0, 1)

# Set max 10 x-ticks for ax2
xticks2 = df['timeStamp'].drop_duplicates()
xticks2 = xticks2.iloc[np.linspace(0, len(xticks2)-1, min(10, len(xticks2))).astype(int)]
ax2.set_xticks(xticks2)
ax2.set_xticklabels(xticks2, rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
