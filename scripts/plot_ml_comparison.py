"""
ML Models Comparison Plotter.

Generates a high-quality bar chart comparing the accuracy of the 
different classical Machine Learning models evaluated for the OCR module.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def generate_comparison_chart() -> None:
    # Model names and their respective accuracies
    models = ['SVM\n(PCA)', 'KNN\n(HOG)', 'LDA\n(Normal Bayes)']
    accuracies = [94.90, 95.87, 97.12]
    
    # Professional color palette (highlighting the best performing model)
    colors = ['#5b9bd5', '#5b9bd5', '#70ad47'] 
    
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(models, accuracies, color=colors, width=0.55)
    
    # Focus the Y-axis to make differences visible
    ax.set_ylim(90, 100)
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title('OCR Model Performance Comparison', fontsize=18, fontweight='bold', pad=20, color='#333333')
    
    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold', color='#333333')
        
    # Clean up the chart (remove top and right borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Add a subtle grid behind the bars
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#eeeeee')
    ax.set_axisbelow(True)
    
    # Customize tick labels
    plt.xticks(fontsize=12, fontweight='bold', color='#555555')
    plt.yticks(fontsize=11, color='#555555')
    
    # Ensure the output directory exists
    os.makedirs('docs/assets', exist_ok=True)
    save_path = 'docs/assets/ml_models_comparison.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] High-resolution chart saved to: {save_path}")
    
    # Display the chart
    plt.show()

if __name__ == '__main__':
    generate_comparison_chart()