import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import io

st.set_page_config(page_title="Global Air Pollution Dashboard", layout="wide")
st.title("🌍 Global Air Pollution Interactive Dashboard ")

# Load the datasets
@st.cache_data
def load_data():
    pollution = pd.read_csv("global_air_pollution_dataset.csv")
    cities = pd.read_csv("worldcities.csv")
    return pollution, cities

pollution, cities = load_data()

# Merge with coordinates
df = pollution.merge(cities, left_on=["City", "Country"], right_on=["city", "country"], how="left")
df = df.dropna(subset=["lat", "lng"])  # Keep valid coordinates


pollutants = ["NO2 AQI Value","Ozone AQI Value","PM2.5 AQI Value","CO AQI Value"]

st.sidebar.header("🔎 Filter Data")
country_filter = st.sidebar.selectbox("Select Country", ["All"] + sorted(df["Country"].unique()))

filtered = df[df["Country"] == country_filter] if country_filter != "All" else df.copy()

# MODEL SELECTION

st.sidebar.subheader("⚙️ Choose Clustering Model")
model_choice = st.sidebar.selectbox("Select Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])

# CLUSTERING

X = filtered[pollutants].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cluster_labels = None

if model_choice == "K-Means":
    k = st.sidebar.slider("Clusters (K)", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)

elif model_choice == "DBSCAN":
    eps = st.sidebar.slider("EPS Radius", 0.1, 5.0, 1.2)
    min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = model.fit_predict(X_scaled)

elif model_choice == "Hierarchical":
    k = st.sidebar.slider("Clusters", 2, 10, 4)
    model = AgglomerativeClustering(n_clusters=k)
    cluster_labels = model.fit_predict(X_scaled)

filtered.loc[X.index, "Cluster"] = cluster_labels

# Silhouette Score
if len(set(cluster_labels)) > 1:
    st.sidebar.success(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.3f}")

# CLUSTER SUMMARY

st.subheader("📊 Cluster Summary Table")
st.dataframe(filtered.groupby("Cluster")[pollutants].mean().round(2))

# VISUAL 1- Pollutant Distribution 

st.subheader("📦 Pollutant Distribution Comparison ")
fig_box = px.box(filtered, y=pollutants, points="all", height=400)
st.plotly_chart(fig_box)

# VISUAL 2 - Pairplot

st.subheader("🔗 Pollutant Correlation ")
fig_matrix = px.scatter_matrix(filtered, dimensions=pollutants, color="Cluster", height=800)
st.plotly_chart(fig_matrix)

# VISUAL 3 — Correlation Heatmap

st.subheader("🔥 Correlation of Pollutants")
corr = filtered[pollutants].corr()
fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
st.plotly_chart(fig_heatmap)

# VISUAL 4 — Cluster Scatter Plot

st.subheader("📌 Cluster Visualization ")
fig_scatter = px.scatter(filtered, x="NO2 AQI Value", y="PM2.5 AQI Value", color="Cluster", hover_data=["City", "Country"], width=900, height=500)
st.plotly_chart(fig_scatter)

# VISUAL 5 — Bar Chart of Average Pollutants 

st.subheader("📊 Average Pollutants ")
cluster_avg = filtered.groupby("Cluster")[pollutants].mean().reset_index()
fig_bar = px.bar(cluster_avg, x="Cluster", y=pollutants, barmode="group", height=450)
st.plotly_chart(fig_bar)

# DOWNLOAD OPTIONS

st.subheader("⬇️ Download Results")

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_data, "pollution_clusters.csv", "text/csv")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    filtered.to_excel(writer, index=False, sheet_name="Clusters")

st.download_button("Download Excel", excel_buffer.getvalue(), "pollution_clusters.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.subheader("📄 Dataset Preview")
st.dataframe(filtered)



