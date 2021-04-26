import nltk
import numpy as np
import pandas as pd
import pathlib

def preprocess(sentence):
    '''Preprocess a given sentence
    '''
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False)
    sentence = ' '.join(tokenizer.tokenize(sentence))
    return sentence

def prepare_quora_dataset(input_path, output_dir, val_size=3000, test_size=30000):
    '''Given the input path to train.csv (from Quora dataset),
    output a file that 
    Args:
        input_path (str): path to train.csv
        output_dir (str): directory to write output files
    '''
    splits = ['train', 'val', 'test']
    
    df = pd.read_csv(input_path)
    duplicates = df[df['is_duplicate']==1]
    
    # preprocess duplicate questions
    duplicates['question1'] = duplicates['question1'].apply(preprocess)
    duplicates['question2'] = duplicates['question2'].apply(preprocess)
    
    test_df = duplicates[-test_size:]
    val_df = duplicates[-test_size - val_size:-test_size]
    train_df = duplicates[:-test_size - val_size]
    
    for split in splits:
        
        split_dir = pathlib.PosixPath(output_dir, split)
        if not split_dir.exists():
            split_dir.mkdir()
        
        orig_path = pathlib.PosixPath(split_dir, 'orig.txt')
        ref_path = pathlib.PosixPath(split_dir, 'ref.txt')
        
        if split == 'train':
            df_split = train_df
        elif split == 'val':
            df_split = val_df
        elif split == 'test':
            df_split = test_df
            
        df_split['question1'].to_csv(orig_path, sep='\n', header=False, index=False)
        df_split['question2'].to_csv(ref_path, sep='\n', header=False, index=False)

if __name__ == '__main__':
    # TODO: use argparse and call prepare_quora_dataset
    pass