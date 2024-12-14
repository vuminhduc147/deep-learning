import torch
import torch.nn as nn
import torch.optim as optim

# Xây dựng mô hình LSTM
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Lấy output từ bước thời gian cuối cùng
        return out

