# 🚀 BDA Final Project Improvement Plan

## Project: Fitbit Fitness Tracker Data Analysis
**Student:** Nguyen Dong Hai — 24072183  
**Reference:** [Fraud Detection Using ML — Full Python Data Science Project](https://www.youtube.com/watch?v=4Od5_z28iIE)

---

## 📋 Current State — What You Already Have

| Area | What You Did | Status |
|------|-------------|--------|
| **Data Loading** | Loaded `dailyActivity_merged.csv` (940 rows, 33 users) | ✅ Done |
| **Column Selection** | Selected 8 key columns (Id, Date, Steps, Activity Minutes, Calories) | ✅ Done |
| **Feature Engineering** | Created `TotalMinutes`, `TotalHours`, `DayOfWeek` | ✅ Basic |
| **Data Cleaning** | Checked nulls & duplicates, converted date types | ✅ Done |
| **EDA Visualizations** | Histogram (day of week), correlation heatmap, scatter plots, pie chart | ✅ Done |
| **Unsupervised ML** | K-Means Clustering → 3 clusters (Low/Moderate/High Activity) | ✅ Done |
| **Anomaly Detection** | Percentile-based (5th/95th) outlier detection on TotalSteps | ✅ Basic |
| **Dashboard** | Streamlit app with 4 pages (Overview, Trends, Segmentation, Anomalies) | ✅ Done |
| **Documentation** | Report (.docx) + Presentation (.pptx) | ✅ Done |

---

## 🔴 What's Missing (Compared to the YouTube Video)

The YouTube video demonstrates a **professional, end-to-end ML pipeline** for classification. Your project is missing these critical components:

### 1. ❌ No Supervised Machine Learning Model
You only have unsupervised clustering. The video builds a **Logistic Regression classifier** with a clear prediction target. You have no predictive model.

### 2. ❌ No Scikit-Learn Pipeline (`Pipeline` + `ColumnTransformer`)
The video uses `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder` inside a clean pipeline. Your preprocessing is done manually step-by-step in loose notebook cells.

### 3. ❌ No Train/Test Split
You never split your data into training and testing sets. This means you cannot properly evaluate any model's performance.

### 4. ❌ No Model Evaluation Metrics
No Accuracy, Precision, Recall, F1-Score, or Confusion Matrix. You cannot quantify how well your analysis performs.

### 5. ❌ No Class Imbalance Handling
The video addresses imbalanced data using `class_weight='balanced'`. You haven't considered this.

### 6. ❌ No Model Comparison
Only K-Means is used. No comparison between multiple algorithms (e.g., Logistic Regression vs. Random Forest vs. Decision Tree).

### 7. ❌ No Model Persistence (Saving/Loading)
You imported `joblib` in `main.py` but never actually save or load a trained model.

### 8. ❌ No Prediction Feature in Dashboard
Your Streamlit dashboard is read-only (displays data). It doesn't let users input values and get predictions.

---

## ✅ Improvement Plan — Step by Step

### Phase 1: Enhanced Feature Engineering (in Notebook)

> **Goal:** Create richer features from the existing data, similar to the video's feature engineering approach.

**New features to create:**
- `ActiveMinutesRatio` = (VeryActiveMinutes + FairlyActiveMinutes) / TotalMinutes
- `StepsPerActiveMinute` = TotalSteps / (VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes)
- `IsWeekend` = 1 if Saturday/Sunday, else 0
- `CaloriesPerStep` = Calories / TotalSteps (handle division by zero)
- `ActivityIntensity` = VeryActiveMinutes × 3 + FairlyActiveMinutes × 2 + LightlyActiveMinutes × 1
- `ReachedGoal` = 1 if TotalSteps ≥ 10,000 (WHO recommendation), else 0 → **This becomes the classification TARGET**

**Files to modify:**
- `fitbit_fitness_tracker_data.ipynb` — add new cells for feature engineering

---

### Phase 2: Build a Supervised ML Classification Model (in Notebook)

> **Goal:** Predict whether a user will reach their daily 10,000-step goal based on activity patterns. This mirrors the video's classification approach.

#### Step 2.1 — Define the Problem
```python
# Create binary target variable
df['ReachedGoal'] = (df['TotalSteps'] >= 10000).astype(int)
```

#### Step 2.2 — Train/Test Split
```python
from sklearn.model_selection import train_test_split

features = ['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes',
            'SedentaryMinutes', 'Calories', 'TotalDistance', 'ActiveMinutesRatio',
            'IsWeekend', 'ActivityIntensity']

X = df[features]
y = df['ReachedGoal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

#### Step 2.3 — Build a Scikit-Learn Pipeline (Key technique from the video!)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(X_train, y_train)
```

#### Step 2.4 — Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Did Not Reach', 'Reached Goal']))

# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Did Not Reach', 'Reached Goal'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — Step Goal Prediction')
plt.show()
```

#### Step 2.5 — Handle Class Imbalance (Key technique from the video!)
- Use `class_weight='balanced'` in Logistic Regression (already included above)
- Alternatively, try SMOTE oversampling for comparison

#### Step 2.6 — Compare Multiple Models
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

# Plot comparison
plt.bar(results.keys(), results.values(), color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
```

**Files to modify:**
- `fitbit_fitness_tracker_data.ipynb` — add new sections for ML pipeline

---

### Phase 3: Save the Best Model with Joblib

> **Goal:** Persist the trained model so the Streamlit dashboard can load and use it for predictions.

```python
import joblib

# Save the best pipeline
joblib.dump(best_pipeline, 'step_goal_model.pkl')

# Also save the updated dataframe
df.to_csv('preprocessed_dailyActivity.csv', index=False)
```

**New files created:**
- `step_goal_model.pkl` — saved trained model

---

### Phase 4: Upgrade the Streamlit Dashboard

> **Goal:** Add two new pages to the dashboard — "Model Performance" and "Predict".

#### New Page: 🎯 Model Performance
Display the model's evaluation results directly in the dashboard:
- Accuracy score as a big metric
- Confusion matrix as a heatmap
- Classification report as a table
- Model comparison bar chart

#### New Page: 🔮 Predict
Let the user input activity values and get a real-time prediction:
```python
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
        total_distance = st.slider("Total Distance (km)", 0.0, 28.0, 5.0)

    # Load model and predict
    model = joblib.load('step_goal_model.pkl')
    # ... compute engineered features and predict
```

**Files to modify:**
- `main.py` — add "Model Performance" and "Predict" pages

---

### Phase 5: Improve Existing Notebook Structure

> **Goal:** Make the notebook more professional and well-organized, similar to the video's clean structure.

- [ ] Add clear **markdown section headers** for each phase:
  - `# 1. Data Loading & Exploration`
  - `# 2. Data Cleaning & Preprocessing`
  - `# 3. Feature Engineering`
  - `# 4. Exploratory Data Analysis (EDA)`
  - `# 5. Unsupervised Learning — K-Means Clustering`
  - `# 6. Supervised Learning — Classification Pipeline`
  - `# 7. Model Evaluation & Comparison`
  - `# 8. Model Export`
- [ ] Add **text explanations** between code cells (what you're doing and why)
- [ ] Remove redundant `df` display calls (cells 8, 10, 12, 14, 19)
- [ ] Fix the typo: column renamed to `Data` should be `Date`

---

## 📁 Final Project Structure (After Improvements)

```
BDA-Final-NguyenDongHai-24072183/
├── dailyActivity_merged.csv            # Raw data
├── preprocessed_dailyActivity.csv      # Cleaned + engineered features + clusters
├── fitbit_fitness_tracker_data.ipynb   # Full analysis notebook (EDA + ML)
├── step_goal_model.pkl                 # NEW: Saved ML model
├── main.py                             # Streamlit dashboard (6 pages now)
├── BDAProjectReport.docx               # Report (update with new findings)
├── NguyenDongHai-24072183-presentation.pptx  # Slides (update)
├── plan.md                             # This improvement plan
└── BDA.xlsx                            # Supporting data
```

---

## 🎯 Priority Summary

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 🔴 High | Add supervised ML model (Logistic Regression + Pipeline) | Medium | Very High |
| 🔴 High | Add train/test split + model evaluation metrics | Low | Very High |
| 🔴 High | Build Scikit-Learn Pipeline with ColumnTransformer | Medium | Very High |
| 🟡 Medium | Handle class imbalance (class_weight='balanced') | Low | High |
| 🟡 Medium | Compare multiple models (LR vs RF vs DT) | Low | High |
| 🟡 Medium | Save model with joblib + add prediction page to dashboard | Medium | High |
| 🟢 Low | Clean up notebook structure & add markdown explanations | Low | Medium |
| 🟢 Low | Add new engineered features | Low | Medium |
| 🟢 Low | Update report & presentation with new results | Medium | Medium |

---

> **Bottom line:** Your project has a solid foundation (EDA, clustering, dashboard). The biggest gap compared to the YouTube video is the **absence of a supervised ML pipeline with proper evaluation**. Adding a classification model with a Scikit-Learn Pipeline, train/test split, and evaluation metrics will take your project from "good" to "professional-grade."
