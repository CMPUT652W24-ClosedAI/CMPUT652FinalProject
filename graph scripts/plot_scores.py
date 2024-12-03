import matplotlib.pyplot as plt
import pandas as pd

def scatter_plot_from_file(title, file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Extract the scores
    asymmetry_scores = data['AsymmetryScore']
    fairness_scores = data['FairnessScore']
    
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(asymmetry_scores, fairness_scores, c='blue', alpha=0.7, label='Data Points')
    plt.xlabel('Asymmetry Score')
    plt.ylabel('Unfairness Score')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    plt.xlim(0, 120)  # Set x-axis range
    plt.ylim(0, 400)  # Set y-axis range
    plt.show()

#scatter_plot_from_file('Random Baseline', 'RANDOM_BASELINE_MANHATTAN_JUDGE.txt')
#scatter_plot_from_file('Manhattan Baseline', 'manhattan2.txt')
scatter_plot_from_file('Value Functions', 'value_functions.txt')

# manhattan: 1733087971 -> similar to random baseline?
# manhattan 2: 1733087971
# manhattan 3: 1733087115

# random_baseline: random_baseline.txt

# value_functions: 1733117438
