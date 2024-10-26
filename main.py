import streamlit as st
import numpy as np
import pickle
import requests
from sklearn.linear_model import Ridge

# Load your trained model (ridge.pkl)
with open('ridge.pkl', 'rb') as file:
    model = pickle.load(file)

# Your OpenWeather API Key
API_KEY = '86f2fcf0ef80738d847825e233078c73'

# Function to get weather data from OpenWeather API
def get_weather_data(city):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind = data['wind']['speed'] * 3.6  # Convert m/s to km/h
        rain = data.get('rain', {}).get('1h', 0)  # Rain in last hour, default to 0 if not present
        return temp, humidity, wind, rain
    else:
        st.error("City not found or API limit reached.")
        return None, None, None, None

# Function to calculate FFMC based on input values
def calculate_ffmc(temp, rh, rain):
    try:
        mo = 147.2 * (101 - rh) / (59.5 + rh)
        rf = rain - 0.5 if rain > 0.5 else 0
        if rf <= 0:
            mr = mo  # Avoid zero division if rf is zero or less
        else:
            mr = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
        
        ffmc = (59.5 * (147.2 - mr)) / (147.2 + mr)
        return max(0, min(ffmc, 101))
    except ZeroDivisionError:
        return 0  # Return 0 if division by zero occurs

def calculate_dmc(temp, rh, rain):
    re = 0 if rain <= 1.5 else 0.92 * rain - 1.27
    dmc = 5 * (rh / temp) + re if temp > 0 else 0  # Avoid division by zero
    return max(0, dmc)  # DMC should not be negative

def calculate_isi(wind_speed, ffmc):
    if ffmc > 0:
        return (wind_speed * ffmc) / 50
    return 0  # Return 0 if ffmc is zero to avoid division by zero

# Streamlit interface
st.title("Fire Weather Index Prediction")

# Input city name
city = st.text_input("Enter City Name for Real-time Weather Data")

# Get weather data from API
if city:
    temp, humidity, wind, rain = get_weather_data(city)

    if temp is not None:
        # Display real-time weather data
        st.write(f"Temperature: {temp} °C")
        st.write(f"Humidity: {humidity} %")
        st.write(f"Wind Speed: {wind} km/h")
        st.write(f"Rain: {rain} mm")

        # Calculate FFMC, DMC, and ISI
        ffmc = calculate_ffmc(temp, humidity, rain)
        dmc = calculate_dmc(temp, humidity, rain)
        isi = calculate_isi(wind, ffmc)

        st.write("Calculated Parameters:")
        st.write(f"FFMC: {ffmc}")
        st.write(f"DMC: {dmc}")
        st.write(f"ISI: {isi}")

        # Prepare data for prediction
        input_data = np.array([[temp, humidity, wind, rain, ffmc, dmc, isi]])

        # Predict FWI using the loaded model
        if st.button("Predict FWI"):
            fwi_prediction = model.predict(input_data)
            st.write(f"Predicted Fire Weather Index (FWI): {fwi_prediction[0]:.2f}")

            # Add cautionary note
            st.warning("Note: This is a predicted value and should not be taken too seriously. Actual FWI values may vary based on numerous factors.")






