import seaborn as sns
import json
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot multipliers from JSON data.")
    parser.add_argument("--file", '-f', required=True, help="Path to the JSON file containing multipliers")
    parser.add_argument("--output", '-o', required=True, help="Path to save the output plot")
    args = parser.parse_args()

    with open(args.file, "r") as file:
        data = json.load(file)

    # Extract the multipliers for plotting
    multipliers = data['best_trial'].get("multipliers", dict())

    # Sort dict alphabetically by keys
    multipliers = [value for key, value in sorted(multipliers.items())]

    sns.boxplot(multipliers)

    # Save the plot to a file
    plt.savefig(args.output)
