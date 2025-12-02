import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Accident Severity Demo", layout="wide")

st.title("Accident Severity Prediction – Interactive Demo 🚦")

# ======================
# Load cleaned dataset
# ======================
import zipfile
import urllib.request
import os

zip_url = "https://raw.githubusercontent.com/<your-username>/<repo-name>/main/US_Accidents_sample100k.zip"
zip_path = "US_Accidents_sample100k.zip"
csv_path = "US_Accidents_sample100k.csv"

if not os.path.exists(csv_path):
    urllib.request.urlretrieve(zip_url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

df = pd.read_csv(csv_path)

# ======================
# Dataset Preview
# ======================
st.header("📘 Dataset Preview")
st.dataframe(df.head())


# ======================
# Filters Section
# ======================
st.header("🔍 Filter the Data")

col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox("Select State", df["State"].unique())

with col2:
    severity = st.selectbox("Select Severity", sorted(df["Severity"].unique()))

with col3:
    weather = st.selectbox("Weather Condition", ["All"] + sorted(df["Weather_Condition"].unique()))


# Apply filters
filtered_df = df[(df["State"] == state) & (df["Severity"] == severity)]

if weather != "All":
    filtered_df = filtered_df[filtered_df["Weather_Condition"] == weather]


st.write("Filtered Results:", len(filtered_df), "rows")
st.dataframe(filtered_df.head())


# ======================
# Map Section
# ======================
st.header("🗺️ Accident Location Map")

map_df = filtered_df[["Start_Lat", "Start_Lng"]].dropna()
map_df = map_df.rename(columns={"Start_Lat": "latitude", "Start_Lng": "longitude"})

st.map(map_df)


# ======================
# Charts Section
# ======================
# ======================
# Filtered Severity Chart (Dynamic)
# ======================
st.header("📊 Severity Distribution (Filtered)")

fig_filt, ax_filt = plt.subplots(figsize=(6,4))  # smaller chart
filtered_df["Severity"].value_counts().sort_index().plot(kind="bar", ax=ax_filt)
ax_filt.set_xlabel("Severity Level")
ax_filt.set_ylabel("Count")
ax_filt.set_title("Severity Distribution for Selected Filters")

st.pyplot(fig_filt)


# ======================
# Global Severity Chart (Static)
# ======================
st.header("🌐 Overall Severity Distribution (Full Dataset)")

fig_all, ax_all = plt.subplots(figsize=(6,4))  # reduced chart size
df["Severity"].value_counts().sort_index().plot(kind="bar", ax=ax_all, color="gray")
ax_all.set_xlabel("Severity Level")
ax_all.set_ylabel("Count")
ax_all.set_title("Global Severity Distribution for Entire Dataset")

st.pyplot(fig_all)


# ======================
# Time of Day Filter
# ======================
st.header("⏰ Accidents by Hour")

selected_hour = st.slider("Select Hour of Day", 0, 23, 12)

hour_df = df[df["Start_Hour"] == selected_hour]

st.write(f"Accidents at {selected_hour}:00 → {len(hour_df)} records")

fig2, ax2 = plt.subplots()
hour_df["Severity"].value_counts().sort_index().plot(kind="bar", ax=ax2)
ax2.set_title("Severity at Selected Hour")
ax2.set_xlabel("Severity")
ax2.set_ylabel("Count")
st.pyplot(fig2)


# ======================
# Model Summary
# ======================
st.header("🤖 Model Summary")

st.write("""
Random Forest performed best among tested models.

**Logistic Regression Accuracy:** ~66–70%  
**Random Forest Accuracy:** ~75–82%  

Random Forest works better because:
- It captures non-linear relationships  
- Handles mixed numerical/categorical data  
- More robust for noisy accident data  
""")
