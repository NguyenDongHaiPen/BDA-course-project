import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
import json
import os

# Page setup
st.set_page_config(page_title="📊 Unified Fitbit Dashboard", layout="wide")

# Load preprocessed dataset
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_dailyActivity.csv")
    if "ActivityDate" in df.columns:
        df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
    elif "Date" in df.columns:
        df["ActivityDate"] = pd.to_datetime(df["Date"])
    
    # Map Cluster values to human-readable labels
    cluster_map = {
        0: "Low Activity",
        1: "Moderate Activity",
        2: "High Activity"
    }
    if "Cluster" in df.columns:
        df["ClusterLabel"] = df["Cluster"].map(cluster_map)
    return df

df = load_data()

# Sidebar
st.sidebar.title("🏃 Fitbit Dashboard")
user_ids = df["Id"].unique()
selected_user = st.sidebar.selectbox("Select User ID", user_ids)
page = st.sidebar.radio("Navigate", ["Overview", "Trends", "Segmentation", "Anomalies", "Model Performance", "Predict"])

df_user = df[df["Id"] == selected_user]

# Overview Page
if page == "Overview":
    st.title("📋 User Activity Overview")
    st.metric("Avg Steps", int(df_user["TotalSteps"].mean()))
    st.metric("Avg Calories Burned", int(df_user["Calories"].mean()))
    st.metric("Total Active Minutes", df_user["VeryActiveMinutes"].sum())

    st.subheader("Activity Breakdown")
    st.bar_chart(df_user[["VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]].sum())

# Trends Page
elif page == "Trends":
    st.title("📈 Daily Activity Trends")

    # Toggle to show combined or separate graphs
    combine = st.checkbox("Combine All Metrics into One Chart", value=False)

    if combine:
        combined_df = df_user[["ActivityDate", "TotalSteps", "Calories", "VeryActiveMinutes"]].copy()
        combined_df = pd.melt(combined_df, id_vars=["ActivityDate"], 
                              value_vars=["TotalSteps", "Calories", "VeryActiveMinutes"],
                              var_name="Metric", value_name="Value")
        fig = px.line(combined_df, x="ActivityDate", y="Value", color="Metric",
                      title="Combined Daily Activity Metrics")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(px.line(df_user, x="ActivityDate", y="TotalSteps", title="Steps Over Time"), use_container_width=True)
        st.plotly_chart(px.line(df_user, x="ActivityDate", y="Calories", title="Calories Burned Over Time"), use_container_width=True)
        st.plotly_chart(px.line(df_user, x="ActivityDate", y="VeryActiveMinutes", title="Very Active Minutes Over Time"), use_container_width=True)


# Segmentation Page
elif page == "Segmentation":
    st.title("🧬 Activity Segmentation")
    if "ClusterLabel" in df.columns:
        st.success("✅ 'ClusterLabel' column created from Cluster. Visualizing user clusters.")
        st.write(df_user[["ActivityDate", "TotalSteps", "Calories", "Cluster", "ClusterLabel"]])

        st.subheader("Cluster Distribution")
        cluster_counts = df["ClusterLabel"].value_counts()
        st.bar_chart(cluster_counts)

        fig = px.scatter(df, x="TotalSteps", y="Calories", color="ClusterLabel",
                         title="User Segmentation by Activity Level",
                         labels={"ClusterLabel": "Activity Cluster"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'Cluster' column not found in dataset.")

# Anomalies Page
elif page == "Anomalies":
    st.title("⚠️ Anomaly Detection (Total Steps)")
    q_low = df_user["TotalSteps"].quantile(0.05)
    q_high = df_user["TotalSteps"].quantile(0.95)
    outliers = df_user[(df_user["TotalSteps"] < q_low) | (df_user["TotalSteps"] > q_high)]
    
    st.write(f"🔍 Found {len(outliers)} anomalies.")
    st.dataframe(outliers[["ActivityDate", "TotalSteps", "Calories"]])

    # ✅ Plot histogram (only once, correctly indented)
    st.subheader("Distribution of Total Steps with IQR Thresholds")
    fig, ax = plt.subplots()
    ax.hist(df_user["TotalSteps"], bins=30, color='skyblue', edgecolor='black')
    ax.axvline(q_low, color='red', linestyle='--', label=f"5% Threshold: {int(q_low)}")
    ax.axvline(q_high, color='green', linestyle='--', label=f"95% Threshold: {int(q_high)}")
    ax.set_title("Histogram of TotalSteps")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

elif page == "Model Performance":
    st.title("🎯 Model Performance")
    
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        
        st.write("### Evaluation Results")
        
        best_model = max(metrics, key=metrics.get)
        best_accuracy = metrics[best_model]
        
        st.metric(f"Best Model Accuracy ({best_model})", f"{best_accuracy:.2%}")
        
        st.write("### Model Comparison")
        fig, ax = plt.subplots()
        ax.bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_title("Model Accuracy Comparison")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    else:
        st.warning("Model metrics not found. Run the notebook first.")
        
elif page == "Predict":
    st.title("🔮 Step Goal Predictor")
    st.write("Enter your daily activity to predict if you'll reach 10,000 steps!")

    col1, col2 = st.columns(2)
    with col1:
        very_active = st.slider("Very Active Minutes", 0, 210, 20)
        fairly_active = st.slider("Fairly Active Minutes", 0, 143, 13)
        lightly_active = st.slider("Lightly Active Minutes", 0, 518, 200)
    with col2:
        sedentary = st.slider("Sedentary Minutes", 0, 1440, 700)
        calories = st.slider("Calories", 0, 4900, 2000)

    day_of_week = st.selectbox("Day of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    if st.button("Predict"):
        if os.path.exists('step_goal_model.pkl'):
            model = joblib.load('step_goal_model.pkl')
            
            is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
            total_minutes = very_active + fairly_active + lightly_active + sedentary
            active_minutes_ratio = (very_active + fairly_active) / total_minutes if total_minutes > 0 else 0
            activity_intensity = very_active * 3 + fairly_active * 2 + lightly_active * 1
            
            # Create feature array
            # Features expected: 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories', 'TotalDistance', 'ActiveMinutesRatio', 'IsWeekend', 'ActivityIntensity'
            input_data = pd.DataFrame([{
                'VeryActiveMinutes': very_active,
                'FairlyActiveMinutes': fairly_active,
                'LightlyActiveMinutes': lightly_active,
                'SedentaryMinutes': sedentary,
                'Calories': calories,
                'ActiveMinutesRatio': active_minutes_ratio,
                'IsWeekend': is_weekend,
                'ActivityIntensity': activity_intensity
            }])
            
            pred = model.predict(input_data)[0]
            if pred == 1:
                st.success("🎉 Prediction: You WILL reach your 10,000 steps goal!")
                st.balloons()
            else:
                st.error("📉 Prediction: You might NOT reach your 10,000 steps goal. Keep moving!")
        else:
            st.warning("Model not found. Run the notebook first to generate the model.")