import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml  # Import the yaml library
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
# from data.data_preparation import load_data, get_data_loaders
from model import ConcatNet
from data.data_loader import load_data
from data.data_preparation import process_data, get_data_loaders


# Load configuration from the YAML file
with open("src/config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data_path, num_epochs=5, batch_size=2000, learning_rate=1e-2, weight_decay=5e-1):
    train_df, valid_df = load_data(data_path, id_val)
    train_df, valid_df = process_data(train_df, valid_df)

    train_loader, valid_loader = get_data_loaders(train_df, valid_df, batch_size)

    # Initialize the model with configuration from YAML
    model = ConcatNet({
        'num_users': config['num_users'],
        'num_items': config['num_items'],
        'emb_size': config['emb_size'],
        'emb_dropout': config['emb_dropout'],
        'fc_layer_sizes': config['fc_layer_sizes'],
        'dropout': config['dropout'],
        'out_range': config['out_range']
    }).to(device)

    criterion = torch.nn.MSELoss(reduction='sum')  # Using sum to compute total loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_loss = np.inf
    train_losses = []
    valid_losses = []
    valid_rmse = []
    valid_mae = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            r_pred = model(u, i)
            loss = criterion(r_pred, r[:, None])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0
        total_abs_error = 0
        total_squared_error = 0
        num_samples = 0
        with torch.no_grad():
            for u, i, r in valid_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                r_pred = model(u, i)
                loss = criterion(r_pred, r[:, None])
                valid_loss += loss.item()
                
                # Accumulate MAE and RMSE
                abs_error = torch.abs(r_pred - r[:, None]).sum().item()
                squared_error = torch.sum((r_pred - r[:, None]) ** 2).item()
                
                total_abs_error += abs_error
                total_squared_error += squared_error
                num_samples += len(r)

        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)

        # Calculate MAE and RMSE
        mae = total_abs_error / num_samples
        rmse = np.sqrt(total_squared_error / num_samples)

        valid_rmse.append(rmse)
        valid_mae.append(mae)

        scheduler.step(valid_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss}, Valid Loss = {valid_loss}, RMSE = {rmse}, MAE = {mae}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = deepcopy(model.state_dict())

    # Load the best model and save it
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), 'saved_models/best_model.pth')

    # Plot the losses and metrics
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.plot(valid_rmse, label='RMSE')
    plt.plot(valid_mae, label='MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

# Call the train_model function
if __name__ == "__main__":
    data_path = "data/ml-100k/"  # Replace with the correct path to your data
    id_val = 1
    train_model(data_path)
