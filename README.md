# 🌍 Global Air Pollution Analysis & Interactive Dashboard

An interactive **Streamlit-based data analysis dashboard** for exploring **global air pollution levels**, clustering world cities using machine learning, and visualizing patterns with dynamic charts and global maps.

🔗 **Live App:**  https://global-airpollution-analysis.streamlit.app/
📁 **Status:** Completed  
📦 **Tech Stack:** Streamlit, Python, Pandas, Plotly, Scikit-Learn

---

## 📌 Project Overview

This project provides an end-to-end solution for analyzing global air quality data.  
It integrates multiple datasets, applies clustering algorithms, and displays insights using an interactive web-based dashboard.

Key goals of the project:

- Analyze global pollutant levels (NO₂, O₃, PM2.5, CO)
- Apply multiple clustering algorithms (K-Means, DBSCAN, Hierarchical)
- Build a dynamic dashboard using Streamlit
- Enable users to interactively explore and download results

---

## 🌟 Features

### 🔎 **Interactive Filters**
- Filter by country  
- Select clustering algorithm  
- Adjust clustering parameters  

### 🤖 **Machine Learning Models**
- K-Means Clustering  
- DBSCAN  
- Agglomerative Clustering  
- Silhouette Score Evaluation  

### 📊 **Visualizations**
- Cluster Summary Table  
- Pollutant Distribution Boxplot  
- Scatter Matrix  
- Correlation Heatmap  
- Cluster Scatter Plot  
- Grouped Bar Charts  

---

## 📸 Screenshot

### **Dashboard Preview Page**
![Dashboard](images/screenshot1.png)

---

## 📂 Project Structure

```
📂 Global-Air-Pollution-Analysis
│── app.py
│── global_air_pollution_dataset.csv
│── worldcities.csv
│── requirements.txt
│── README.md
│── images/
      └── screenshot1.png
```

---

## ⚙️ Installation & Running Locally

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🚀 Deployment

The project is deployed using **Streamlit Cloud**.

To deploy your own version:
1. Upload all project files to GitHub  
2. Add `requirements.txt`  
3. Select the repository in Streamlit Cloud → Deploy  
4. Set `app.py` as the main module

---

## 🧠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming |
| **Streamlit** | Web app framework |
| **Pandas** | Data cleaning & manipulation |
| **Plotly** | Interactive data visualizations |
| **Scikit-Learn** | Clustering algorithms |

---

## 👩‍💻 Author

**VIMALI VINCY M**  
Passionate about data analysis, machine learning, and building interactive dashboards.

---

