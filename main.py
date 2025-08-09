
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Upload file CSV
st.title("📊 Sales Data Analysis")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Hiển thị dữ liệu
    st.subheader("📄 Data Preview")
    st.dataframe(data.head())

    # Thông tin chung
    st.subheader("ℹ️ Data Info")
    st.write(data.shape)
    st.write(data.dtypes)

    # Xử lý dữ liệu thiếu
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        data[col].fillna(data[col].median(), inplace=True)
    for col in categorical_cols:
        data[col].fillna('Unknown', inplace=True)

    # Thống kê mô tả
    st.subheader("📈 Statistical Summary")
    st.write(data.describe(include='all'))

    # Vẽ biểu đồ
    st.subheader("📊 Numeric Distributions")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    # Ma trận tương quan
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
