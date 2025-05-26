# Imports necessary modules
import argparse
import sys
import csv
import json
import gzip
import nltk

nltk.download("punkt_tab")
nltk.download("punkt")

csv.field_size_limit(sys.maxsize)

if __name__ == "__main__":
	# Defines argument parser with 2 arguments: corpus csv in, tokenized out
	parser = argparse.ArgumentParser()
	parser.add_argument("json_in", help="original data in json format")
	parser.add_argument("tokenized_jsonl", help="zipped json file to write into")
	args = parser.parse_args()

	# Opens  arguments using with, csv for reading, tokenized out for gzip writing
	with gzip.open(args.json_in, "rt") as json_in, gzip.open(args.tokenized_jsonl,"wt") as json_out:
		count = 0

		# Iterates over each row of dict reader
		for line in json_in:

			# Defines a csv DictReader with the target object being csv file opened
			row = json.loads(line)

			print(count)
			count += 1

			row["full_text"] = " ".join(row["full_text"])
			row["full_text"] = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(row["full_text"].lower())]

			# Writes new version of dictionary to the gzip outfile
			json_out.write(json.dumps(row) + "\n")
