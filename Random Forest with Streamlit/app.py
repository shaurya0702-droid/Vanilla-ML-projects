import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Student Exam Score Predictor")
st.write("Random Forest Regression with Tree Visualization")

# Inputs
hours = st.slider("Hours Studied", 0.0, 12.0, 6.0)
attendance = st.slider("Attendance (%)", 50, 100, 80)
sleep = st.slider("Sleep Hours", 4.0, 9.0, 7.0)
previous = st.slider("Previous Exam Score", 0, 100, 65)

if st.button("Predict Score"):
    X = np.array([[hours, attendance, sleep, previous]])
    prediction = model.predict(X)
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")

st.subheader("🌳 One Decision Tree from Random Forest")

tree = model.estimators_[0]

fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    tree,
    max_depth=3,
    feature_names=[
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Score"
    ],
    filled=True,
    ax=ax
)

st.pyplot(fig)
