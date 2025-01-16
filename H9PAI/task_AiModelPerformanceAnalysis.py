import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Generate data using NumPy
models = [f'Model_{i}' for i in range(1,6)]
accuracy = np.random.uniform(0.8, 1.0, size=5)
precision = np.random.uniform(0.7, 1.0, size=5)
recall = np.random.uniform(0.6, 1.0, size=5)
training_time = np.random.randint(10, 50, size=5)

# Create the DataFrame
data = {
    'Model' : models,
    'Accuracy' : accuracy,
    'Precision' : precision,
    'Recall' : recall,
    'Training_time' : training_time,
}
df = pd.DataFrame(data)

# Analyze the data
avg_metrics = df[['Accuracy', 'Precision', 'Recall', 'Training_time']].mean()
max_metrics = df[['Accuracy', 'Precision', 'Recall', 'Training_time']].max() 
min_metrics = df[['Accuracy', 'Precision', 'Recall', 'Training_time']].min()
best_model = df.loc[df['Accuracy'].idxmax()]

# Visualize the data

# Bar chart for accuracy, precision, recall
x = np.arange(len(models))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - bar_width, df['Accuracy'], bar_width, label='Accuracy', color='skyblue')
ax.bar(x, df['Precision'], bar_width, label='Precision', color='orange')
ax.bar(x+bar_width, df['Recall'], bar_width, label='Recall', color='yellow')

ax.set_xlabel('Models')
ax.set_ylabel('Metrics')
ax.set_title('Model Performance: Accuracy, Precision, Recall')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Adding labels to each bar
for i, (acc, prec, rec) in enumerate(zip(df['Accuracy'], df['Precision'], df['Recall'])):
    ax.text(i-bar_width, acc+0.01, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(i, prec+0.01, f'{prec:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(i+bar_width, rec+0.01, f'{rec:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Line chart for Training Time
plt.figure(figsize=(8, 5))
plt.plot(models, df['Training_time'], marker='*', linestyle='-', color='red', label='Trainig Time')
plt.title('Training Time for Each Model')
plt.xlabel('Models')
plt.ylabel('Training Time (Minutes)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print(avg_metrics, max_metrics, min_metrics, best_model)
