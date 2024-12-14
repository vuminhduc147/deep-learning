import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

def process_and_split_data(filename):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(filename)

    print("Dữ liệu ban đầu:")
    print(df.head())

    # Xử lý dữ liệu thiếu
    df = df.dropna()
    if df.empty:
        print("\nLỗi: Dữ liệu sau khi loại bỏ giá trị thiếu bị trống. Kiểm tra lại file CSV.")
        return

    print("\nDữ liệu sau khi loại bỏ giá trị thiếu:")
    print(df.head())

    # Chuyển cột 'date' thành kiểu datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

        # Tạo các đặc trưng từ cột 'date'
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday

        # Xóa cột 'date' sau khi tạo các đặc trưng
        df = df.drop(columns=['date'])

    # Mã hóa dữ liệu phân loại cho cột 'city'
    if 'city' in df.columns:
        df = pd.get_dummies(df, columns=['city'], drop_first=True)

    # Tách dữ liệu đầu vào (X) và đầu ra (y)
    X = df.drop(columns=['temp_c'])  # Dữ liệu đầu vào
    y = df['temp_c']  # Dữ liệu đầu ra (nhiệt độ)

    # Kiểm tra dữ liệu đầu vào và đầu ra
    if X.empty or y.empty:
        print("\nLỗi: Dữ liệu đầu vào hoặc đầu ra bị trống.")
        return

    # Chuẩn hóa dữ liệu đầu vào
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Chia dữ liệu thành tập huấn luyện, kiểm tra và kiểm định
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Tạo thư mục 'data' nếu chưa có
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # Lưu dữ liệu vào các file CSV trong folder 'data'
    train_data = pd.DataFrame(X_train, columns=X.columns)
    train_data['temp_c'] = y_train.values
    train_data.to_csv('./data/train.csv', index=False)

    val_data = pd.DataFrame(X_val, columns=X.columns)
    val_data['temp_c'] = y_val.values
    val_data.to_csv('./data/validation.csv', index=False)

    test_data = pd.DataFrame(X_test, columns=X.columns)
    test_data['temp_c'] = y_test.values
    test_data.to_csv('./data/test.csv', index=False)

    print("\nDữ liệu đã được chuẩn hóa và lưu vào các file trong thư mục 'data': train.csv, validation.csv, test.csv")

if __name__ == "__main__":
    input_file = './data/weather_data_13_12_2023_to_12_12_2024.csv'

    try:
        process_and_split_data(input_file)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file.")
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
