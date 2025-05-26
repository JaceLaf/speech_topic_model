# Imports necessary modules
import argparse
import gzip
import json
import random
import nltk
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
import gensim.parsing.preprocessing as gpp

# Adds each argument using argparse with defaults for each variable
parser = argparse.ArgumentParser()
parser.add_argument(
    "data",
    help="Data file model will be trained on."
)
parser.add_argument(
    "subdocs",
    help="File to save subdocuments to."
)
parser.add_argument(
    "dictionary",
    help="File to save the dictionary to."
)
parser.add_argument(
    "--subdocument_length",
    dest="subdocument_length",
    default=200,
    type=int,
    help="The number of tokens to have in each subdocument."
)
parser.add_argument(
    "--minimum_word_length",
    dest="minimum_word_length",
    default=3,
    type=int,
    help="Minimum word length."
)
parser.add_argument(
    "--maximum_subdocuments",
    dest="maximum_subdocuments",
    type=int,
    help="Maximum number of subdocuments to use."
)
parser.add_argument(
    "--minimum_word_count",
    dest="minimum_word_count",
    default=30,
    type=int
)
parser.add_argument(
    "--maximum_word_proportion",
    dest="maximum_word_proportion",
    default=0.7,
    type=float,
    help="Maximum proportion of subdocuments a word can occur in before considering it 'too common'."
)
parser.add_argument(
    "--chunksize",
    dest="chunksize",
    default=2000,
    type=int,
    help="How many subdocuments to consider 'at a time': this affects how much the model 'jumps around' during training."
)

args = parser.parse_args()

subdocuments = []

# Opens tokenized data file
with gzip.open(args.data, "rt") as ifd:

    # Reads in and strips each row
    for row in ifd:
        jrow = json.loads(row)
        tokens = " ".join([t for s in jrow["full_text"] for t in s])
        tokens = gpp.strip_non_alphanum(tokens)
        tokens = gpp.remove_stopwords(tokens)
        tokens = gpp.split_on_space(tokens)

        num_subdocuments = int(len(tokens) / args.subdocument_length)

        # Appends each filtered word back into subdocument list
        for subnum in range(num_subdocuments):
            start_token_index = subnum * args.subdocument_length
            end_token_index = (subnum + 1) * args.subdocument_length
            subdocument_tokens = tokens[start_token_index:end_token_index]
            subdocuments.append(subdocument_tokens)

# Filters subdocument list based on inputted maximum
random.shuffle(subdocuments)
subdocuments = subdocuments[0:args.maximum_subdocuments if args.maximum_subdocuments else len(subdocuments)]

# Creates a dictionary for training based on subdocuments
dct = Dictionary(documents=subdocuments)
dct.filter_extremes(no_below=args.minimum_word_count, no_above=args.maximum_word_proportion)

# Writes results to files
with gzip.open(args.subdocs, "wt") as out_subdocs:
    json.dump(subdocuments, out_subdocs)

dct.save(args.dictionary)
