import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Data Preprocessing
def preprocess_data(data):
    # Fill missing values for 'Age' with the median
    imputer = SimpleImputer(strategy='median')
    data['Age'] = imputer.fit_transform(data[['Age']])

    # Fill missing values for 'Embarked' with the most frequent value
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Fill missing 'Fare' in test data with median
    if 'Fare' in data.columns:
        data['Fare'] = imputer.fit_transform(data[['Fare']])

    # Convert 'Sex' and 'Embarked' to numeric using Label Encoding
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

    # Drop columns not useful for prediction
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    return data

# Preprocess train and test data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Separate features and target variable from train data
X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_data['Survived']

# Split train data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
val_predictions = model.predict(X_val)
print("Accuracy on validation data:", accuracy_score(y_val, val_predictions))
print(classification_report(y_val, val_predictions))

# Prepare test data for predictions
X_test = test_data.drop(['PassengerId'], axis=1)
test_predictions = model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": test_predictions
})

# Save the predictions to a CSV file
submission.to_csv("submission.csv", index=False)

print("Prediction complete. Submission file 'submission.csv' created.")
