import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Data
red_wine = pd.read_csv('winequality/winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('winequality/winequality-white.csv', delimiter=';')

# Transform Response Variable
def transform_quality(quality):
    if quality <= 4:
        return 0
    elif quality <= 6:
        return 1
    else:
        return 2

red_wine['quality'] = red_wine['quality'].apply(transform_quality)
white_wine['quality'] = white_wine['quality'].apply(transform_quality)

# EDA
red_quality_counts = red_wine['quality'].value_counts()
white_quality_counts = white_wine['quality'].value_counts()

red_means = red_wine.groupby('quality').mean()
white_means = white_wine.groupby('quality').mean()

# Split Data
trainNegre, testNegre = train_test_split(red_wine, test_size=0.3, random_state=42)
trainBlanc, testBlanc = train_test_split(white_wine, test_size=0.3, random_state=42)

# Model Training
rf_model_red = RandomForestClassifier(random_state=42)
rf_model_red.fit(trainNegre.drop('quality', axis=1), trainNegre['quality'])

rf_model_white = RandomForestClassifier(random_state=42)
rf_model_white.fit(trainBlanc.drop('quality', axis=1), trainBlanc['quality'])

# Model Evaluation
def evaluate_model(model, test_data, label):
    predictions = model.predict(test_data.drop('quality', axis=1))
    accuracy = accuracy_score(test_data['quality'], predictions)
    conf_matrix = confusion_matrix(test_data['quality'], predictions)
    class_report = classification_report(test_data['quality'], predictions)
    
    print(f"{label} Model Accuracy: {accuracy}")
    print(f"{label} Model Confusion Matrix:\n{conf_matrix}")
    print(f"{label} Model Classification Report:\n{class_report}")

evaluate_model(rf_model_red, testNegre, "Red Wine")
evaluate_model(rf_model_white, testBlanc, "White Wine")

# Predictions
sample_red = pd.DataFrame({
    'fixed acidity': [7.4],
    'volatile acidity': [1.015],
    'citric acid': [0.01],
    'residual sugar': [2.9],
    'chlorides': [0.075],
    'free sulfur dioxide': [4],
    'total sulfur dioxide': [11],
    'density': [0.896],
    'pH': [3.43],
    'sulphates': [0.5],
    'alcohol': [11.01]
})

sample_white = pd.DataFrame({
    'fixed acidity': [5.7],
    'volatile acidity': [0.19],
    'citric acid': [0.28],
    'residual sugar': [10.6],
    'chlorides': [0.028],
    'free sulfur dioxide': [48],
    'total sulfur dioxide': [134],
    'density': [0.894],
    'pH': [3.02],
    'sulphates': [0.49],
    'alcohol': [11.03]
})

red_prediction = rf_model_red.predict(sample_red)
white_prediction = rf_model_white.predict(sample_white)

print(f"Predicted quality of red wine sample: {red_prediction[0]}")
print(f"Predicted quality of white wine sample: {white_prediction[0]}")
