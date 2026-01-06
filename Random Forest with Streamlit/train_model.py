import pandas as pd
import pickle  #for serializing (saving) and deserializing (loading) python objects
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("student_data.csv")
X = df[["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Score"]]
y = df["Final_Exam_Score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor( # setting hyperparameters
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:   #save model
    pickle.dump(model, f)
print("Model trained and saved as model.pkl")
