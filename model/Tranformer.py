import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Định nghĩa mô hình Transformer
class WeatherTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads):
        super(WeatherTransformer, self).__init__()
        
        # Transformer Encoder Layer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_size,  # Kích thước embedding
            nhead=num_heads,  # Số lượng attention heads
            dim_feedforward=hidden_size  # Kích thước của hidden layer trong feed-forward
        )
        
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        
        # Lớp fully connected để dự đoán đầu ra
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Chuẩn bị dữ liệu cho transformer (batch_first=True)
        x = x.permute(1, 0, 2)  # Chuyển đổi chiều để phù hợp với đầu vào của Transformer (seq_len, batch_size, features)
        x = self.transformer(x)  # Pass qua transformer encoder
        x = x[-1, :, :]  # Lấy output tại time step cuối cùng
        x = self.fc(x)  # Pass qua FC layer để dự đoán giá trị
        return x
