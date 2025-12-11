import matplotlib.pyplot as plt

# Data
models = ['GPT-2-1.5B\n(Teacher)', 'GPT-2-120M\n(Student)', 'GPT-2-120M\n(Student) w/ prompt']
perplexity = [77.41, 36.54, 33.71]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars = ax.bar(models, perplexity, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Customize chart
ax.set_title('Perplexity of Student with Soft Prompts¹²', fontsize=14, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=12)

# Add value labels on bars
for bar, value in zip(bars, perplexity):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{value}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()