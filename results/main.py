import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")

# Clean and convert percentage columns to numeric
data['Ts Accuracy'] = data['Ts Accuracy'].str.rstrip('%').astype(float)
data['F1 Score'] = data['F1 Score'].str.rstrip('%').astype(float)
data['AUC-ROC'] = data['AUC-ROC'].str.rstrip('%').astype(float)
data['Score'] = (data['Ts Accuracy']/2 + data['F1 Score'] + data['AUC-ROC'])/2.5
data = data.sort_values('Score')

# Set a smaller figure size
plt.figure(figsize=(8, 10))

# Define bar width
bar_width = 0.6

# Generate bar positions
indices = np.arange(len(data))

# Plot stacked horizontal bar chart
plt.barh(indices, data['Ts Accuracy'], bar_width, label='Test Accuracy', color='steelblue')
plt.barh(indices, data['F1 Score'], bar_width, left=data['Ts Accuracy'], label='F1 Score', color='orange')
plt.barh(indices, data['AUC-ROC'], bar_width, left=data['Ts Accuracy'] + data['F1 Score'], label='AUC-ROC', color='green')
plt.barh(indices, data['Score'], bar_width, left=data['Ts Accuracy'] + data['F1 Score'] + data['AUC-ROC'], label='Score', color='purple')

# Add labels and title
plt.ylabel('Model Architecture', fontsize=10)
plt.xlabel('Scores (%)', fontsize=10)
plt.title('Stacked Scores for Different Architectures', fontsize=14)
plt.yticks(indices, data['Architecture'], fontsize=9)

# Add legend
plt.legend()

# Adjust layout to avoid clipping of labels
plt.tight_layout()

# Save the plot to a file
plt.savefig('stacked_scores_chart.png', dpi=300, bbox_inches='tight')

# Show the plot (optional)
plt.show()
