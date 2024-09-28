# src/data/data_loader.py

import pandas as pd

def load_data(data_path, id_val=1):
    """Load train and validation data from the specified path."""
    train_df = pd.read_csv(f'{data_path}u{id_val}.base', sep='\t', header=None)
    train_df.columns = ['user_id', 'item_id', 'rating', 'ts']
    
    valid_df = pd.read_csv(f'{data_path}u{id_val}.test', sep='\t', header=None)
    valid_df.columns = ['user_id', 'item_id', 'rating', 'ts']

    return train_df, valid_df
