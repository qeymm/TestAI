import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample training data
# Features: [Age, Gender(0 = Male, 1 = Female), Country (0 = Indonesia, 1 = USA), Insurance Type(0 = Health, 1 = Life)]
X_train = np.array([
    [52, 0, 0, 0],
    [30, 1, 1, 1],
    [45, 0, 0, 0],
    [25, 1, 1, 1],
    [60, 0, 0, 0]
])

# Target (labels): Best Plan (0 = Basic, 1 = Silver, 2 = Gold)
y_train = np.array([0, 2, 1, 2, 0])

# Train a simple RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'insurance_model.pkl')

print("Model saved successfully!")
