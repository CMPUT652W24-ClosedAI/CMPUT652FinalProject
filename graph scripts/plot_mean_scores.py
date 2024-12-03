import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_means_from_files(file_dict):
    """
    Plots the mean AsymmetryScore and FairnessScore for each file in the given dictionary.

    Parameters:
        file_dict (dict): A dictionary where keys are file names (for labeling) and values are file paths.
    """
    # Prepare lists to store file names and their corresponding mean scores
    labels = []
    mean_asymmetry_scores = []
    mean_fairness_scores = []

    # Process each file
    for file_name, file_path in file_dict.items():
        data = pd.read_csv(file_path)
        mean_asymmetry = data['AsymmetryScore'].mean()
        mean_fairness = data['FairnessScore'].mean()

        # Append results
        labels.append(file_name)
        mean_asymmetry_scores.append(mean_asymmetry)
        mean_fairness_scores.append(mean_fairness)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    for label, x, y in zip(labels, mean_asymmetry_scores, mean_fairness_scores):
        plt.scatter(x, y, s=100, label=label)

    # Add labels, legend, and grid
    plt.xlabel('Mean Asymmetry Score')
    plt.ylabel('Mean Unfairness Score')
    plt.title('Mean Scores for Each Experiment')
    plt.grid(True)
    plt.legend()
    plt.show()

file_dict = {
    'Random Baseline': 'RANDOM_BASELINE_MANHATTAN_JUDGE.txt',
    'Value Functions': 'value_functions.txt',
    'Manhattan Baseline': 'manhattan3.txt',
    '0': '0.txt'
}
file_dict = {
    '0.0': 'vf00.txt',
    '0.25': 'vf025.txt',
    '0.5': 'vf05.txt',
    '0.8': 'vf08.txt',
    '1.0': 'vf10.txt',
}
plot_means_from_files(file_dict)
