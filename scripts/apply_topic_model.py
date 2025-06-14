# Imports necessary modules
import gzip
import re
import logging
import random
import pickle
import json
import argparse

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

import gensim.parsing.preprocessing as gpp


logging.basicConfig(level=logging.INFO)

# Uses argparse to load in file paths and needed info
parser = argparse.ArgumentParser()

parser.add_argument(
    "model",
    help="Model file generated by training script."
)
parser.add_argument(
    "subdoc",
    help="Data file model will be applied to."
)
parser.add_argument(
    "dictionary",
    help="Dictionary file that trains model"
)
parser.add_argument(
    "counts",
    help="Counts output file."
)
parser.add_argument(
    "--group_min",
    dest="group_min",
    default=1500,
    type=int,
    help="The minimum value of the group field (any documents lower than this will be discarded)."
)
parser.add_argument(
    "--group_resolution",
    dest="group_resolution",
    default=10,
    type=int,
    help="The size of each group (e.g. number of years, or whatever units 'group_field' uses)."
)

args = parser.parse_args()

topic_counts = {}

# Opens each module and dictionary
with open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())

dct = Dictionary.load(args.dictionary)

# Opens subdocuments
with gzip.open(args.subdoc, "rt") as sub:
    subdocuments = json.load(sub)

    # Sums each topic's prevalence in the subdocuments based on the model
    for doc in subdocuments:
        bow = dct.doc2bow(doc)
        topic_dist = model.get_document_topics(bow)

        for topic_id, topic_prob in topic_dist:
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + topic_prob


# Writes counts to output file
with open(args.counts, "wt") as ofd:
    ofd.write(
        json.dumps(
            [(k, v) for k, v in sorted(topic_counts.items())],
            indent=4
        )
    )
