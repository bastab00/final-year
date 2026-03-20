# File: model_comparison_chart.py

import matplotlib.pyplot as plt
import numpy as np

# Model names

models = ["Logistic Regression", "Linear SVM", "SBERT + SVM", "Random Forest", "XGBoost"]

# Metrics

accuracy = [0.877, 0.921, 0.903, 0.922, 0.912]
f1_class1 = [0.73, 0.80, 0.78, 0.78, 0.77]
f1_macro = [0.82, 0.87, 0.86, 0.86, 0.86]

x = np.arange(len(models))  # label locations
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(10,6))

# Bars

rects1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='#4C72B0')
rects2 = ax.bar(x, f1_class1, width, label='F1 (Class 1)', color='#55A868')
rects3 = ax.bar(x + width, f1_macro, width, label='F1 (Macro)', color='#C44E52')

# Labels and titles

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha='right')
ax.set_ylim(0, 1)
ax.legend()

# Annotate bars

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
ax.annotate(f'{height:.2f}',
xy=(rect.get_x() + rect.get_width()/2, height),
xytext=(0,3),
textcoords="offset points",
ha='center', va='bottom')

plt.tight_layout()
plt.show()
