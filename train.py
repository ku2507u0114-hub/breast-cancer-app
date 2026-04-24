import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# 🔹 Load dataset
df = pd.read_csv("data/breast_cancer.csv")

# 🔥 Remove unnecessary columns (VERY IMPORTANT)
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# Remove unnamed columns (if any)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 🔹 Convert target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 🔹 Features & target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Prediction
y_pred = model.predict(X_test)

# 🔹 Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation mean:", cv_scores.mean())
print("Cross-validation std:", cv_scores.std())

# 🔹 Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("\n✅ Model trained successfully!")