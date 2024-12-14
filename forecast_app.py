import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model  # Đảm bảo load_model được import
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import streamlit as st

# Đọc dữ liệu train, validation và test
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')  # Chuyển đổi cột date
    return data

# Chuẩn hóa dữ liệu
def normalize_data(data):
    scaler = MinMaxScaler()  # Hoặc có thể sử dụng StandardScaler
    features = ['temp_c', 'humidity', 'rainfall_mm', 'wind_speed_kph']  # Các cột cần chuẩn hóa
    data_scaled = scaler.fit_transform(data[features])  # Chuẩn hóa dữ liệu
    return data_scaled, scaler  # Trả về dữ liệu đã chuẩn hóa và đối tượng scaler để phục hồi dữ liệu

# Hàm tạo chuỗi thời gian
def create_sequences(data_scaled, target, sequence_length):
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])  # Dữ liệu đầu vào
        y.append(target[i+sequence_length])  # Dữ liệu mục tiêu
    return np.array(X), np.array(y)

# Xây dựng mô hình LSTM
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(units=1)  # Dự đoán một giá trị
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Hàm tạo dự báo từ mô hình
def forecast_with_model(model, city_data, scaler, days=7, sequence_length=30):
    # Chuẩn bị đầu vào cho mô hình (dữ liệu gần nhất)
    recent_data = city_data[['temp_c', 'humidity', 'rainfall_mm', 'wind_speed_kph']].tail(sequence_length).values
    recent_data_scaled = scaler.transform(recent_data)  # Chuẩn hóa dữ liệu đầu vào
    
    recent_data_scaled = recent_data_scaled.reshape(1, sequence_length, recent_data_scaled.shape[1])  # Reshape cho LSTM
    
    # Tạo dự báo
    forecast = model.predict(recent_data_scaled)
    forecast_temps = forecast[0, :days]  # Giả sử dự báo chỉ trả về nhiệt độ

    # Tạo ngày tương lai
    future_dates = pd.date_range(start=city_data['date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    forecast_data = pd.DataFrame({
        'date': future_dates,
        'temp_c': forecast_temps
    })
    return forecast_data

# Hàm hiển thị dữ liệu và dự báo
def display_city_weather(data, city_name, model, scaler, sequence_length=30):
    # Lọc dữ liệu cho thành phố được chọn
    city_data = data[data['city'] == city_name]
    
    # Trực quan hóa dữ liệu thời tiết
    st.subheader(f"Thời tiết tại {city_name}")
    st.write(city_data)
    
    # Biểu đồ nhiệt độ theo ngày
    plt.figure(figsize=(10, 5))
    plt.plot(city_data['date'], city_data['temp_c'], marker='o', label="Nhiệt độ (°C)")
    plt.xlabel("Ngày")
    plt.ylabel("Nhiệt độ (°C)")
    plt.title(f"Nhiệt độ tại {city_name}")
    plt.legend()
    st.pyplot(plt)
    
    # Dự báo thời tiết
    st.subheader(f"Dự báo thời tiết 7 ngày tới tại {city_name}")
    forecast_data = forecast_with_model(model, city_data, scaler)
    st.write(forecast_data)
    
    # Biểu đồ dự báo
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_data['date'], forecast_data['temp_c'], marker='o', linestyle='--', color='orange', label="Dự báo nhiệt độ (°C)")
    plt.xlabel("Ngày")
    plt.ylabel("Nhiệt độ (°C)")
    plt.title(f"Dự báo nhiệt độ tại {city_name}")
    plt.legend()
    st.pyplot(plt)

    # Thông tin tổng quan
    avg_temp = city_data['temp_c'].mean()
    total_rainfall = city_data['rainfall_mm'].sum()
    st.write(f"Nhiệt độ trung bình: {avg_temp:.2f}°C")
    st.write(f"Tổng lượng mưa: {total_rainfall:.2f} mm")

# Giao diện Streamlit
st.title("Thông tin thời tiết và dự báo")

# Tải dữ liệu từ file CSV
uploaded_file = st.file_uploader("Tải lên file CSV thời tiết", type="csv")
model_path = st.text_input("Nhập đường dẫn tới mô hình học sâu đã lưu (.h5):")

if uploaded_file and model_path:
    # Đọc dữ liệu
    data = load_and_process_data(uploaded_file)
    
    # Chuẩn hóa dữ liệu
    data_scaled, scaler = normalize_data(data)
    
    # Tạo chuỗi dữ liệu cho train, validation, và test
    sequence_length = 30  # Số ngày để tạo chuỗi
    X_data, y_data = create_sequences(data_scaled, data['temp_c'].values, sequence_length)
    
    # Tải mô hình
    try:
        model = load_model(model_path)
        st.success("Mô hình học sâu đã được tải thành công!")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()
    
    # Hiển thị các thành phố có trong dữ liệu
    cities = data['city'].unique()
    city_choice = st.selectbox("Chọn thành phố", cities)
    
    # Hiển thị thông tin thời tiết và dự báo
    display_city_weather(data, city_choice, model, scaler)

# Lệnh để chạy Streamlit: 
# python -m streamlit run forecast_app.py