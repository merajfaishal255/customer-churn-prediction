import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load data
data = pd.read_csv("data.csv")

# Convert target to numbers
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

# 2. Features & target
X = data[["Tenure", "MonthlyCharges", "SupportCalls"]]
y = data["Churn"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as churn_model.pkl")
