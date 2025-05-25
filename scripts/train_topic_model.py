import csv
import re
import logging
import random
import pickle
import argparse
import gzip
import json
import nltk
from nltk.corpus import stopwords

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy
import gensim.parsing.preprocessing as gpp

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "dictionary",
    help="Dictionary the model should be trained from"
)
parser.add_argument(
    "subdocs",
    help="File containing all the subdocuments"
)
parser.add_argument(
    "model",
    help="File to save the resulting model to."
)
parser.add_argument(
    "results",
    help="File to save the outputted topics to."
)
parser.add_argument(
    "--num_topics",
    dest="num_topics",
    default=10,
    type=int,
    help="Number of topics."
)
parser.add_argument(
    "--chunksize",
    dest="chunksize",
    default=2000,
    type=int,
    help="How many subdocuments to consider 'at a time': this affects how much the model 'jumps around' during training."
)
parser.add_argument(
    "--passes",
    dest="passes",
    default=10,
    type=int,
    help="How many times to 'pass over' the data during training."
)
parser.add_argument(
    "--iterations",
    dest="iterations",
    default=20,
    type=int,
    help="A highly-technical parameter for the training process (see the GenSim documentation if interested)."
)
parser.add_argument(
    "--random_seed",
    dest="random_seed",
    type=int,
    help="If this is set to a number, the training should produce the same model when given the same data and parameters."
)
args = parser.parse_args()

if args.random_seed:
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)

with gzip.open(args.subdocs, "rt") as sub:
    subdocuments = json.load(sub)

dct = Dictionary.load(args.dictionary)

temp = dct[0]

train = [dct.doc2bow(subdoc) for subdoc in subdocuments]

model = LdaModel(
    train,
    num_topics=args.num_topics,
    id2word=dct,
    alpha="auto",
    eta="auto",
    iterations=args.iterations,
    passes=args.passes,
    eval_every=None,
    chunksize=args.chunksize
)

topics = [
    {"topic_id": int(topic_id), "topic_text": topic_text}
    for topic_id, topic_text in model.print_topics()
]

with open(args.results, "w") as out:
    json.dump(topics, out)

with open(args.model, "wb") as ofd:
    ofd.write(pickle.dumps(model))
