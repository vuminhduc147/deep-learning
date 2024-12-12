import requests
from datetime import datetime, timedelta
import csv
import time

# WeatherAPI key
api_key = '3485c96404d149d4a7f141419241212'

# Hàm lấy dữ liệu thời tiết từ API
def fetch_weather_data(location, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f'http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date_str}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'forecast' in data and 'forecastday' in data['forecast']:
            return data
        else:
            print(f"Error: Unexpected data format for {location} on {date_str}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {location} on {date_str}: {e}")
        return None

# Hàm ghi dữ liệu vào CSV
def write_to_csv(data, filename="weather_data.csv"):
    file_exists = False
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            file_exists = True
    except FileNotFoundError:
        pass  
    

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Ghi tiêu đề cột chỉ một lần
            writer.writerow(["city", "date", "temp_c", "humidity", "rainfall_mm", "wind_speed_kph"])
        for entry in data:
            writer.writerow([entry['city'], entry['date'], entry['temp'], 
                             entry['humidity'], entry['rainfall'], entry['wind_speed']])
    print(f"Weather data has been written to {filename}")

# Hàm thu thập dữ liệu thời tiết
def collect_weather_data(cities, start_date, end_date):
    current_date = start_date
    weather_data = []
    
    while current_date <= end_date:
        for city in cities:
            data = fetch_weather_data(city, current_date)
            if data:
                forecast = data['forecast']['forecastday'][0]['day']
                weather_data.append({
                    'city': city,
                    'date': current_date.strftime("%d/%m/%Y"),
                    'temp': forecast['avgtemp_c'],
                    'humidity': forecast['avghumidity'],
                    'rainfall': forecast['totalprecip_mm'],
                    'wind_speed': forecast['maxwind_kph']
                })
            time.sleep(1)  
        current_date += timedelta(days=1)
    
    return weather_data

# Chương trình chính
if __name__ == "__main__":
    try:
        cities = ['Hanoi', 'Ho Chi Minh City', 'Da Nang']  
        
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)  # 1 year
        
        filename = f"weather_data_{start_date.strftime('%d_%m_%Y')}_to_{end_date.strftime('%d_%m_%Y')}.csv"
        
        weather_data = collect_weather_data(cities, start_date, end_date)
        write_to_csv(weather_data, filename)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
