import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import uuid

# Set style and custom color palette
sns.set(style="whitegrid")
colors = ["#2aaa2a", "#d62728"]
sns.set_palette(colors)
custom_palette = {"SOA": "#2aaa2a", "Microservices": "#d62728"}

# File path
file_path = r"C:\Users\Enes\Desktop\test\step_logs.jsonl"

# Read JSONL file in chunks
chunk_size = 10000
chunks = pd.read_json(file_path, lines=True, chunksize=chunk_size)

df_list = []
for chunk in chunks:
    chunk = pd.concat([chunk.drop(['metrics'], axis=1), chunk['metrics'].apply(pd.Series)], axis=1)
    df_list.append(chunk)
df = pd.concat(df_list, ignore_index=True)

# Convert timestamp
df['ts'] = pd.to_datetime(df['ts'], unit='s')

# Generate and save each chart
# 1-3: Line Plots
for metric in ['latency', 'energy', 'throughput']:
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='step', y=metric, hue='choice', style='choice', markers=True, palette=custom_palette)
    plt.title(f'{metric.capitalize()} over Steps by Choice')
    plt.xlabel('Step')
    plt.ylabel(metric.capitalize())
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='SOA', markerfacecolor='#2aaa2a', markersize=10),
                       Line2D([0], [0], marker='s', color='w', label='Microservices', markerfacecolor='#d62728', markersize=10)]
    plt.legend(handles=legend_elements, title='Architecture', loc='upper right')
    plt.savefig(f"{metric}_line_plot.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# 4: Box Plot
plt.figure(figsize=(12, 12))
for i, metric in enumerate(['latency', 'energy', 'throughput'], 1):
    plt.subplot(3, 1, i)
    sns.boxplot(data=df, x='choice', y=metric, hue='choice', palette=custom_palette, legend=False)
    plt.title(f'Distribution of {metric.capitalize()} by Choice')
    plt.xlabel('Architecture')
    plt.ylabel(metric.capitalize())
plt.tight_layout()
plt.savefig("box_plot_metrics.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# 5-6: Violin Plots
for metric in ['latency', 'energy']:
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, x='choice', y=metric, hue='choice', palette=custom_palette, legend=False)
    plt.title(f'{metric.capitalize()} Distribution by Choice')
    plt.xlabel('Architecture')
    plt.ylabel(metric.capitalize())
    plt.savefig(f"{metric}_violin_plot.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# 7: Scatter Plot
plt.figure(figsize=(12, 8))
df_sample = df.sample(frac=0.1, random_state=42)
sns.scatterplot(data=df_sample, x='latency', y='throughput', hue='choice', style='choice', palette=custom_palette)
plt.title('Latency vs. Throughput by Choice')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput')
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='s', color='w', label='SOA', markerfacecolor='#2aaa2a', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Microservices', markerfacecolor='#d62728', markersize=10)]
plt.legend(handles=legend_elements, title='Architecture', loc='upper right')
plt.savefig("latency_vs_throughput_scatter.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# 8: Hexbin Plots
for choice in df['choice'].unique():
    plt.figure(figsize=(12, 8))
    subset = df[df['choice'] == choice]
    
    # Set color dynamically based on choice
    if choice == 'SOA':
        cmap = plt.cm.Greens  # Green color map
    else:  # 'Microservices'
        cmap = plt.cm.Reds  # Red color map
    
    hb = plt.hexbin(subset['energy'], subset['throughput'], gridsize=50, cmap=cmap, mincnt=1)
    plt.title(f'Energy vs. Throughput Density for {choice}')
    plt.xlabel('Energy')
    plt.ylabel('Throughput')
    plt.colorbar(hb, label='Count')
    plt.savefig(f"energy_vs_throughput_{choice.lower()}_hexbin.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


# 9: Bar Plot
plt.figure(figsize=(12, 8))
mean_metrics = df.groupby('choice')[['latency', 'energy', 'throughput']].mean().reset_index()
mean_metrics_melted = mean_metrics.melt(id_vars='choice', var_name='Metric', value_name='Value')
sns.barplot(data=mean_metrics_melted, x='Metric', y='Value', hue='choice', palette=custom_palette)
plt.title('Average Metrics by Choice')
plt.xlabel('Metric')
plt.ylabel('Average Value')
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='s', color='w', label='SOA', markerfacecolor='#2aaa2a', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Microservices', markerfacecolor='#d62728', markersize=10)]
plt.legend(handles=legend_elements, title='Architecture', loc='upper right')
plt.savefig("average_metrics_bar.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# 10-11: Histograms
for metric in ['latency', 'throughput']:
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x=metric, hue='choice', element='step', palette=custom_palette)
    plt.title(f'{metric.capitalize()} Distribution by Choice')
    plt.xlabel(metric.capitalize())
    plt.ylabel('Count')
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='SOA', markerfacecolor='#2aaa2a', markersize=10),
                       Line2D([0], [0], marker='s', color='w', label='Microservices', markerfacecolor='#d62728', markersize=10)]
    plt.legend(handles=legend_elements, title='Architecture', loc='upper right')
    plt.savefig(f"{metric}_histogram.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# 12: Heatmap
for choice in df['choice'].unique():
    plt.figure(figsize=(8, 6))
    corr = df[df['choice'] == choice][['latency', 'energy', 'throughput']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Matrix for {choice}')
    plt.savefig(f"correlation_heatmap_{choice.lower()}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

print("Visualizations saved as individual PDF files:")
print("1. latency_line_plot.pdf")
print("2. energy_line_plot.pdf")
print("3. throughput_line_plot.pdf")
print("4. box_plot_metrics.pdf")
print("5. latency_violin_plot.pdf")
print("6. energy_violin_plot.pdf")
print("7. latency_vs_throughput_scatter.pdf")
print("8. energy_vs_throughput_soa_hexbin.pdf")
print("9. energy_vs_throughput_microservices_hexbin.pdf")
print("10. average_metrics_bar.pdf")
print("11. latency_histogram.pdf")
print("12. throughput_histogram.pdf")
print("13. correlation_heatmap_soa.pdf")
print("14. correlation_heatmap_microservices.pdf")
