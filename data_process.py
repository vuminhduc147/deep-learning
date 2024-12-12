import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys

sys.stdout.reconfigure(encoding='utf-8')

def process_and_split_data(filename):
    """
    Xử lý và chuẩn hóa dữ liệu, sau đó chia thành tập huấn luyện, kiểm tra và kiểm định.
    
    Args:
        filename (str): Tên file CSV chứa dữ liệu thu thập được.

    Outputs:
        Lưu các file train.csv, validation.csv, và test.csv.
    """
    # 1. Đọc dữ liệu từ file CSV
    df = pd.read_csv(filename)

    print("Dữ liệu ban đầu:")
    print(df.head())

    # 2. Kiểm tra dữ liệu thiếu và loại bỏ các hàng có giá trị thiếu
    df = df.dropna()
    if df.empty:
        print("\nLỗi: Dữ liệu sau khi loại bỏ giá trị thiếu bị trống. Kiểm tra lại file CSV.")
        return

    print("\nDữ liệu sau khi loại bỏ giá trị thiếu:")
    print(df.head())

    # 3. Mã hóa dữ liệu phân loại (One-hot encoding cho cột 'city')
    if 'city' in df.columns:
        df = pd.get_dummies(df, columns=['city'], drop_first=True)

    # 4. Tách dữ liệu thành đầu vào (X) và đầu ra (y)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])  # Loại bỏ cột ngày nếu không cần thiết

    X = df.drop(columns=['temp_c'])  # Dữ liệu đầu vào
    y = df['temp_c']  # Dữ liệu đầu ra (mục tiêu dự đoán)

    # 5. Chuẩn hóa dữ liệu đầu vào
    if X.empty or y.empty:
        print("\nLỗi: Dữ liệu đầu vào hoặc đầu ra bị trống.")
        return

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Chia dữ liệu thành tập huấn luyện, kiểm tra, và kiểm định
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 7. Lưu các tập dữ liệu vào file CSV
    train_data = pd.DataFrame(X_train, columns=X.columns)
    train_data['temp_c'] = y_train.values
    train_data.to_csv('train.csv', index=False)

    val_data = pd.DataFrame(X_val, columns=X.columns)
    val_data['temp_c'] = y_val.values
    val_data.to_csv('validation.csv', index=False)

    test_data = pd.DataFrame(X_test, columns=X.columns)
    test_data['temp_c'] = y_test.values
    test_data.to_csv('test.csv', index=False)

    print("\nDữ liệu đã được chuẩn hóa và lưu vào các file: train.csv, validation.csv, test.csv")

if __name__ == "__main__":
    input_file = 'weather_data_13_12_2023_to_12_12_2024.csv'

    try:
        process_and_split_data(input_file)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file.")
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
