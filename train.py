import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model.lstm import WeatherLSTM
from model.Tranformer import WeatherTransformer
import json
from data_loader import data_loader, data_process

# Tải file cấu hình từ JSON
def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def train_model():
    # Tải các tham số cấu hình
    config = load_config()

    # Lấy các tham số từ cấu hình
    model_name = config["model"]["name"]
    input_size = config["model"]["input_size"]
    hidden_size = config["model"]["hidden_size"]
    output_size = config["model"]["output_size"]
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    file_path = config["data"]["file_path"]
    target_columns = config["data"]["target_columns"]

    global model, model_path
    if model_name == 0:
        model = WeatherLSTM(input_size, hidden_size, output_size)
        model_path = "./model/LSTM.h5"
    else:
        model = WeatherTransformer(input_size, hidden_size, output_size, num_layers, num_heads)
        model_path = "./model/Transformer.h5"
        
    # model = WeatherLSTM(input_size, hidden_size, output_size)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_loader, y_train_loader = data_loader('./data/train_data.csv', target_columns=['max', 'min'])
    # X_test_loader, y_test_loader = data_loader('test_data.csv', target_columns=['max', 'min'])

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_train_loader)

        # Tính toán loss
        loss = criterion(y_pred, y_train_loader)

        # Backward pass và cập nhật trọng số
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), model_path)
        
if __name__ == "__main__":
    train_model()
    
