import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Harga Rumah King County", page_icon="🏠")

st.title("🏠 Prediksi Harga Rumah King County")
st.write("Aplikasi Machine Learning untuk memprediksi harga rumah di King County.")

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pth')

data_loaded = load_model()
model = data_loaded['model']
scaler = data_loaded['scaler']

st.subheader("📋 Masukkan Spesifikasi Rumah:")
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1.0, max_value=8.0, value=2.0)
    sqft_living = st.number_input("Luas Bangunan (sqft)", min_value=300, max_value=10000, value=1800)
    floors = st.number_input("Jumlah Lantai", min_value=1.0, max_value=3.5, value=1.0)

with col2:
    waterfront = st.selectbox("Pemandangan Danau", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
    view = st.slider("Kualitas View (0-4)", 0, 4, 0)
    condition = st.slider("Kondisi Rumah (1-5)", 1, 5, 3)
    grade = st.slider("Grade Konstruksi (1-13)", 1, 13, 7)

if st.button("🔮 Hitung Estimasi Harga"):
    # Cek berapa jumlah fitur yang dibutuhkan oleh scaler/model Anda saat ini
    try:
        n_features = scaler.n_features_in_
    except AttributeError:
        n_features = 8  # Default jika atribut tidak ditemukan

    # Menyesuaikan input dengan jumlah fitur yang diminta model
    if n_features == 8:
        input_data = [[bedrooms, bathrooms, sqft_living, floors, waterfront, view, condition, grade]]
    else:
        # Jika model butuh lebih banyak fitur (misal hasil dari OneHotEncoding), 
        # kita lengkapi sisanya dengan nilai 0 agar tidak error
        base_input = [bedrooms, bathrooms, sqft_living, floors, waterfront, view, condition, grade]
        input_data = [base_input + [0] * (n_features - len(base_input))]

    # Transform dan Prediksi
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"💰 **Estimasi Harga Rumah:** ${prediction:,.2f}")
