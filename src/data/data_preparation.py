# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, Dataset

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Dataset Class
# class CollabDataset(Dataset):
#     def __init__(self, df, user_col=0, item_col=1, rating_col=2):  # Corrected the constructor
#         self.df = df
#         self.user_tensor = torch.tensor(self.df.iloc[:, user_col], dtype=torch.long, device=device)
#         self.item_tensor = torch.tensor(self.df.iloc[:, item_col], dtype=torch.long, device=device)
#         self.target_tensor = torch.tensor(self.df.iloc[:, rating_col], dtype=torch.float32, device=device)
        
#     def __getitem__(self, index):  # Corrected method
#         return (self.user_tensor[index], self.item_tensor[index], self.target_tensor[index])

#     def __len__(self):  # Corrected method
#         return self.user_tensor.shape[0]

# def load_data(data_path, id_val=1):
#     # Load train and validation data
#     train_df = pd.read_csv(f'{data_path}u{id_val}.base', sep='\t', header=None)
#     train_df.columns = ['user_id', 'item_id', 'rating', 'ts']
#     train_df['user_id'] = train_df['user_id'] - 1
#     train_df['item_id'] = train_df['item_id'] - 1

#     valid_df = pd.read_csv(f'{data_path}u{id_val}.test', sep='\t', header=None)
#     valid_df.columns = ['user_id', 'item_id', 'rating', 'ts']
#     valid_df['user_id'] = valid_df['user_id'] - 1
#     valid_df['item_id'] = valid_df['item_id'] - 1

#     return train_df, valid_df

# def get_data_loaders(train_df, valid_df, batch_size=2000):
#     train_dataset = CollabDataset(train_df)
#     valid_dataset = CollabDataset(valid_df)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     return train_loader, valid_loader


# src/data/data_preparation.py

import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CollabDataset(Dataset):
    def __init__(self, df):
        self.user_tensor = torch.tensor(df['user_id'].values, dtype=torch.long, device=device)
        self.item_tensor = torch.tensor(df['item_id'].values, dtype=torch.long, device=device)
        self.target_tensor = torch.tensor(df['rating'].values, dtype=torch.float32, device=device)

    def __getitem__(self, index):
        return (self.user_tensor[index], self.item_tensor[index], self.target_tensor[index])

    def __len__(self):
        return len(self.user_tensor)

def process_data(train_df, valid_df):
    """Process the data by adjusting user_id and item_id."""
    train_df['user_id'] -= 1
    train_df['item_id'] -= 1
    valid_df['user_id'] -= 1
    valid_df['item_id'] -= 1
    
    return train_df, valid_df

def get_data_loaders(train_df, valid_df, batch_size=2000):
    """Get DataLoader for training and validation datasets."""
    train_dataset = CollabDataset(train_df)
    valid_dataset = CollabDataset(valid_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader
