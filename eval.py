# evaluate.py
import torch
from model.lstm import WeatherLSTM
from model.Tranformer import WeatherTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import json
from data_loader import data_loader

def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def evaluate_model():
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

    X_test_loader, y_test_loader = data_loader('./data/test_data.csv', target_columns=['max', 'min'])

    # Tải mô hình đã huấn luyện
    if model_name == 0:
        model = WeatherLSTM(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load('./model/LSTM.h5'))  # Đường dẫn đến mô hình đã lưu
    else:
        model = WeatherTransformer(input_size, hidden_size, output_size, num_layers, num_heads)
        model.load_state_dict(torch.load('./model/Transformer.h5'))  # Đường dẫn đến mô hình đã lưu
        
    model.eval()  # Đặt mô hình vào chế độ đánh giá
    # Dự đoán từ mô hình
    with torch.no_grad():
        predictions = model(X_test_loader)

    # Chuyển đổi các tensor thành mảng numpy để tính toán chỉ số đánh giá
    predictions = predictions.numpy()
    test_labels = y_test_loader.numpy()

    # Tính toán Mean Squared Error (MSE) và Root Mean Squared Error (RMSE)
    mae = mean_absolute_error(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(test_labels, predictions)

    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Mean Asolute Error: {mae:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R2 score: {r2:.4f}')
    

if __name__ == "__main__":
    evaluate_model()
