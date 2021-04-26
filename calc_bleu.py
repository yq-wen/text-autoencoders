import nltk
import numpy as np
import pandas as pd
import pathlib
import argparse


def calc_bleu(input_path, rec_path):
    '''Calculates the average bleu score between the original sentence
    and the recreated sentence.
    Arguments:
        input_path: path to the input file
        rec_path: path to the recreated file
    '''
    input_sents = []
    rec_sents = []
    
    with open(input_path) as f_input:
        for line in f_input:
            input_sents.append(line)
    with open(rec_path) as f_rec:
        for line in f_rec:
            rec_sents.append(line)
    
    assert(len(input_sents) == len(rec_sents))
    
    N = len(input_sents)
    bleus = np.zeros((N,))
    
    for i in range(N):
        ref = input_sents[i].split()
        hyp = rec_sents[i].split()
        bleu = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
        bleus[i] = bleu

    return bleus.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_1', help='path to the file of sentences')
    parser.add_argument('path_2', help='path to another file of sentences')

    args = parser.parse_args()

    print(calc_bleu(args.path_1, args.path_2))
