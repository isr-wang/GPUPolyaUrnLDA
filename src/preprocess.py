#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from string import punctuation
import argparse
import random
import struct


STOPWORDS_FNAME = 'stoplists/en.txt'


def preprocess(input_fname, output_fname, remove_stopwords, K):
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
    corpus_tokens = Counter()
    corpus = []

    # Read input file for the first time to count token frequency
    with open(input_fname) as input_file:
        for line in input_file:
            doc_tokens_raw = line.split()

            if len(doc_tokens_raw) <= 2:
                raise SystemExit('Each line must contain document name, document number, and space-separated tokens. Exiting immediately.')

            doc_tokens = []
            for token in doc_tokens_raw[2:]:
                # Lowercase token and strip leading and trailing punctuation
                lowercase_token = token.lower().strip(punctuation)
                # Only add the token if the token is alphabetic and not in the
                # stopwords list
                if lowercase_token.isalpha():
                    if not remove_stopwords:
                        corpus_tokens[lowercase_token] += 1
                        doc_tokens.append(lowercase_token)
                    elif remove_stopwords and lowercase_token not in stopwords:
                        corpus_tokens[lowercase_token] += 1
                        doc_tokens.append(lowercase_token)
            corpus.append(doc_tokens)

    # Sort the corpus tokens by frequency, and assign them each an id
    token_ids = {}
    for idx, token in enumerate(corpus_tokens.most_common()):
        token_ids[token[0]] = idx

    # Open the z, w, and d files to write the topic indicators, document
    # tokens, and document lengths respectively
    z_file = open('{}.z.bin'.format(output_fname), 'wb')
    w_file = open('{}.w.bin'.format(output_fname), 'wb')
    d_file = open('{}.d.bin'.format(output_fname), 'wb')

    w_tot = 0
    # For each document in the corpus,
    for document in corpus:
        # Write the topic indicator z drawn from a Uniform Random Distribution
        z = random.uniform(0, K)
        z_file.write(struct.pack('I', z))
        # Write the document length d for the warp sampler
        d = len(document)
        d_file.write(struct.pack('I', d))
        # turn the tokens into ids using token_ids, and sort them (equivalent
        # to sorting the tokens by their frequency in the corpus)
        w = sorted([token_ids[token] for token in document])
        for token in w:
            w_file.write(struct.pack('I', token))
        print(w, d)
        w_tot += d

    # Close our files
    z_file.close()
    w_file.close()
    d_file.close()

    print('Wrote {0} topic indicators to {1}.z.bin'.format(len(corpus),
                                                           output_fname))
    print('Wrote {0} document tokens to {1}.w.bin'.format(w_tot, output_fname))
    print('Wrote {0} document lengths to {1}.d.bin'.format(len(corpus),
                                                           output_fname))


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
    parser.add_argument('-k', action='store', dest='K', default=10,
                        help='Upper bound of Uniform Random Distribution from which z is drawn')

    args = parser.parse_args()
    preprocess(input_fname=args.input_file,
               output_fname=args.output_file,
               remove_stopwords=args.remove_stopwords,
               K=args.K)
