import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "results_json",
    help="evaluation of each model in JSON format"
)
parser.add_argument(
    "chart",
    help="file to output the results to in rmd format"
)

args = parser.parse_args()

with open(args.results_json, "r") as i:
    results = json.load(i)

with open(args.chart, "w") as o:
    o.write("| Model File    | Coherence  | Perplexity |\n")
    o.write("|---------------|------------|------------|\n")

    for line in results:
        model = line['model_file']
        coherence = f"{line['coherence']:.4f}"
        perplexity = f"{line['perplexity']:.4f}"
        o.write(f"| {model:<13} | {coherence:>10} | {perplexity:>10} |\n")
