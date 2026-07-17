# 🏃 Fitbit Fitness Tracker Data Analysis & ML Dashboard

**Author:** Nguyen Dong Hai (24072183)  
**Course:** Big Data Analytics (BDA)

An end-to-end Machine Learning project analyzing Fitbit fitness tracker data. This project features a robust Python/Scikit-Learn pipeline for predictive modeling and user segmentation, wrapped in an interactive Streamlit dashboard that visualizes activity trends, detects anomalies, and predicts if users will hit their daily step goals.

## 🌟 Key Features

*   **Exploratory Data Analysis (EDA):** Comprehensive analysis of user habits, daily trends, and activity breakdowns.
*   **User Segmentation:** Unsupervised learning (K-Means Clustering) to categorize users into *Low*, *Moderate*, and *High* activity groups.
*   **Predictive Modeling (Supervised ML):** A Scikit-Learn classification pipeline using Logistic Regression (with class imbalance handling) to predict if a user will achieve their daily goal of 10,000 steps.
*   **Interactive Streamlit Dashboard:** 
    *   **Overview:** High-level metrics and activity breakdown.
    *   **Trends:** Interactive time-series charts of steps, calories, and active minutes.
    *   **Segmentation:** Visualizes K-Means clustering distributions.
    *   **Anomalies:** Detects statistical outliers in daily steps.
    *   **Model Performance:** Real-time metrics comparing model accuracies.
    *   **Predict:** Live step-goal prediction using custom user inputs!

## 📁 Project Structure

```
.
├── dailyActivity_merged.csv            # Raw dataset
├── preprocessed_dailyActivity.csv      # Cleaned data with engineered features & clusters
├── fitbit_fitness_tracker_data.ipynb   # Complete ML Pipeline & EDA Notebook
├── step_goal_model.pkl                 # Pre-trained Scikit-Learn model
├── main.py                             # Streamlit dashboard application
├── requirements.txt                    # Project dependencies
└── metrics.json                        # Model evaluation metrics
```

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NguyenDongHaiPen/BDA-course-project.git
   cd BDA-course-project
   ```

2. **Install the dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run main.py
   ```
   The dashboard will automatically open in your default web browser!

## 🛠️ Built With
*   **Python** (Pandas, NumPy)
*   **Scikit-Learn** (Pipelines, Logistic Regression, Random Forest, K-Means)
*   **Plotly & Matplotlib** (Data Visualization)
*   **Streamlit** (Web Application Framework)
