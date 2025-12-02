import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import urllib.request
import os

st.set_page_config(page_title="Accident Severity Interactive Demo", layout="wide")

st.title("🚗 Accident Severity Prediction — Interactive Demo")

# --------------------------
# Load dataset from GitHub ZIP
# --------------------------

zip_url = "https://raw.githubusercontent.com/<your-username>/<repo>/main/US_Accidents_cleaned_100k.zip"
zip_path = "US_Accidents_cleaned_100k.zip"
csv_path = "US_Accidents_cleaned_100k.csv"

if not os.path.exists(csv_path):
    st.info("Downloading dataset... Please wait ⏳")
    urllib.request.urlretrieve(zip_url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

# Load CSV
df = pd.read_csv(csv_path)

st.success("Dataset Loaded Successfully! ✔")

# --------------------------
# Sidebar Filters
# --------------------------

st.sidebar.header("🔍 Filter Options")

states = df["State"].dropna().unique()
state = st.sidebar.selectbox("Select State", states)

months = sorted(df["Start_Month"].dropna().unique())
month = st.sidebar.selectbox("Select Month", months)

filtered_df = df[(df["State"] == state) & (df["Start_Month"] == month)]

st.write(f"### Showing results for: **{state} — Month {month}**")
st.write(f"Total Records: **{len(filtered_df)}**")

# --------------------------
# Section 1 — Quick Overview
# --------------------------

st.header("📊 Overview of Accident Patterns")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Severity Distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="Severity", data=filtered_df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Accidents by Hour of Day")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(filtered_df["Start_Hour"], bins=24, kde=False, ax=ax)
    st.pyplot(fig)

# --------------------------
# Section 2 — Map
# --------------------------

st.header("🗺️ Accident Locations Map")

if "Start_Lat" in filtered_df.columns and "Start_Lng" in filtered_df.columns:
    st.map(filtered_df[["Start_Lat", "Start_Lng"]].dropna())

# --------------------------
# Section 3 — Weather Impact
# --------------------------

st.header("🌧️ Weather Conditions Impact")

weather_counts = filtered_df["Weather_Condition"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=weather_counts.index, y=weather_counts.values, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# --------------------------
# Section 4 — Summary
# --------------------------

st.header("📝 Summary Insights")

st.write("""
- Severe accidents (Severity 4) are **less common** but still present.
- Most accidents occur during **rush hours** (7–9 AM & 4–6 PM).
- Certain weather conditions (Rain, Fog, Snow) show **higher accident counts**.
- Urban states (NY, CA) have **dense accident clusters** visible on the map.
""")
