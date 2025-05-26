# Imports necessary modules
import argparse
import pickle
import json
import gzip
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Uses argparse to take needed file inputs
parser = argparse.ArgumentParser()
parser.add_argument(
    "output",
    help="json evaluation file output"
)
parser.add_argument(
    "subdocs",
    help="file containing subdocuments"
)
parser.add_argument(
    "dictionary",
    help="dictionary model is trained on"
)
parser.add_argument(
    "models",
    nargs="+",
    help="filenames for each of the models"
)

args = parser.parse_args()

# Loads in subdocuments and dictionary
with gzip.open(args.subdocs, "rt") as sub:
    subdocuments = json.load(sub)

dct = Dictionary.load(args.dictionary)

train = [dct.doc2bow(subdoc) for subdoc in subdocuments]

results = []

# Iterates through each model
for file in args.models:
    with open(file, "rb") as f:
        model = pickle.load(f)

        # Calculates coherence score and perplexity
        coherence_model = CoherenceModel (
            model = model,
            texts=subdocuments,
            dictionary=dct,
            coherence='u_mass'
        )
        coherence = coherence_model.get_coherence()

        perplexity = model.log_perplexity(train)

        # Appends results to dictionary
        results.append({
            "model_file": file,
            "coherence": coherence,
            "perplexity": perplexity
        })

# Writes results to file
with open(args.output, "w") as out:
    json.dump(results, out)
