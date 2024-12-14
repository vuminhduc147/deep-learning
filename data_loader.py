import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def data_process():

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('./data/weather.csv')
    # Xem các cột dữ liệu
    print(df.columns)
    # Loại bỏ các feature không quan trọng
    df = df.drop(columns=['province', 'wind_d', 'cloud', 'pressure'])
    # Xem lại các cột dữ liệu sau khi loại bỏ
    print(df.columns)
    # Chuyển ngày tháng thành dạng số (Ngày tính từ mốc thời gian)
    df['date'] = (pd.to_datetime(df['date']) - pd.to_datetime(df['date']).min()).dt.days
    
    # Tiêu chuẩn hóa các cột số
    features = ['max', 'min', 'wind', 'rain', 'humidi', 'date']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Tách dữ liệu thành features và target
    X = df[features].values
    y = df[['max', 'min']].values

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape)  # Kiểm tra kích thước dữ liệu
    X_train_df = pd.DataFrame(X_train, columns=['max', 'min', 'wind', 'rain', 'humidity', 'date'])  # Điều chỉnh tên cột nếu cần
    y_train_df = pd.DataFrame(y_train, columns=['max', 'min'])  # Tùy chỉnh theo các target cần dự báo

    X_test_df = pd.DataFrame(X_test, columns=['max', 'min', 'wind', 'rain', 'humidity', 'date'])  # Điều chỉnh tên cột nếu cần
    y_test_df = pd.DataFrame(y_test, columns=['max', 'min'])  # Tùy chỉnh theo các target cần dự báo

    # Lưu tập huấn luyện vào file CSV
    train_df = pd.concat([X_train_df, y_train_df], axis=1)
    train_df.to_csv('./data/train_data.csv', index=False)

    # Lưu tập kiểm tra vào file CSV
    test_df = pd.concat([X_test_df, y_test_df], axis=1)
    test_df.to_csv('./data/test_data.csv', index=False)

    print("Dữ liệu đã được lưu vào 'train_data.csv' và 'test_data.csv'")


def data_loader(file_path, target_columns):
    # Đọc file CSV
    df = pd.read_csv(file_path)
    
    # Tách feature và target
    X = df.drop(columns=target_columns).values  # Loại bỏ cột target để lấy features
    y = df[target_columns].values  # Lấy cột target

    # Chuẩn hóa các feature (Optional)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chuyển thành tensor (optional)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor

if __name__ == "__main__":
    data_process()
