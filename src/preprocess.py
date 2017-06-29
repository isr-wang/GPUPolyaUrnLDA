#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import Counter


STOPWORDS_FNAME = 'stoplists/en.txt'


def preprocess(input_fname, output_fname, remove_stopwords):
    # Build the stopword list if necessary
    stopwords = set()
    if remove_stopwords:
        with open(STOPWORDS_FNAME) as stopwords_file:
            for line in stopwords_file:
                stopwords.add(line.strip())
        print('Read {0} stopwords from {1}'.format(len(stopwords),
                                                   STOPWORDS_FNAME))

    # Initialize hashtable so that we can count token occurrence. We will be
    # storing the tokens for a document sorted by their frequency in the
    # corpus.
    tokens = Counter()
    corpus = []

    # Read input file for the first time to count token frequency
    with open(input_fname) as input_file:
        for line in input_file:
            doc_tokens_raw = line.split()

            if len(doc_tokens_raw) <= 2:
                raise SystemExit('Line must contain document name, document number, and space-separated tokens. Exiting immediately.')

            doc_tokens = []
            for token in doc_tokens_raw[2:]:
                lowercase_token = token.lower()
                # Only add the token if the token is not in the stopwords list
                if remove_stopwords and lowercase_token not in stopwords:
                    tokens[lowercase_token] += 1
                    doc_tokens.append(lowercase_token)
            corpus.append(doc_tokens)

    print(tokens)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Convert an text file containing one document per line to a binary file')
    parser.add_argument('--input-file', '-i', dest='input_file',
                        action='store', required=True,
                        help='Input file containing one document per line with space separated tokens')
    parser.add_argument('--output-file', '-o', dest='output_file',
                        action='store', required=True,
                        help='Binary files consisting of topic indicators, tokens, and document lengths')
    parser.add_argument('--remove-stopwords', action='store_const', const=True,
                        help='Remove common adverbs, conjunctions, prepositions, pronouns, and such')

    args = parser.parse_args()
    preprocess(input_fname=args.input_file,
               output_fname=args.output_file,
               remove_stopwords=args.remove_stopwords)
