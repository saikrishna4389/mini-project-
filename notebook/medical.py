import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load training data
dataset = pd.read_csv("data/Training.csv")

X = dataset.drop("prognosis", axis=1)
y = dataset["prognosis"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=20)

# Train final model (SVC)
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Save model and label encoder
pickle.dump(svc, open("models/svc.pkl", 'wb'))
pickle.dump(le, open("models/label_encoder.pkl", 'wb'))

# Load support files
precautions = pd.read_csv("data/precautions_df.csv")
workout = pd.read_csv("data/workout_df.csv")
description = pd.read_csv("data/description.csv")
medications = pd.read_csv("data/medications.csv")
diets = pd.read_csv("data/diets.csv")

# Define symptoms and disease mapping
symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}

# Load label encoder
label_encoder = pickle.load(open("models/label_encoder.pkl", 'rb'))

# Prediction function
def get_predicted_value(user_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in user_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    predicted_index = svc.predict([input_vector])[0]
    return label_encoder.inverse_transform([predicted_index])[0]

# Clean list helper
def clean_list(x):
    return [str(i) for i in x if pd.notnull(i)]

# Recommendation helper
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'].dropna().values)
    pre = clean_list(precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten())

    # Parse Medication
    med = medications[medications['Disease'] == dis]['Medication'].values
    med = clean_list(ast.literal_eval(med[0])) if len(med) > 0 else []

    # Parse Diet
    die = diets[diets['Disease'] == dis]['Diet'].values
    die = clean_list(ast.literal_eval(die[0])) if len(die) > 0 else []

    wrkout = clean_list(workout[workout['disease'] == dis]['workout'].values)

    return desc, pre, med, die, wrkout

# CLI testing
if __name__ == "__main__":
    symptoms = input("Enter your symptoms (comma-separated): ")
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    predicted_disease = get_predicted_value(user_symptoms)
    desc, pre, med, die, wrkout = helper(predicted_disease)

    print("\n🩺 Predicted Disease:", predicted_disease)
    print("\n📜 Description:", desc)

    print("\n⚠️ Precautions:")
    for i, item in enumerate(pre, 1):
        print(f"{i}. {item}")

    print("\n💊 Medications:")
    for i, item in enumerate(med, 1):
        print(f"{i}. {item}")

    print("\n🏋️ Workouts:")
    for i, item in enumerate(wrkout, 1):
        print(f"{i}. {item}")

    print("\n🍽️ Diets:")
    for i, item in enumerate(die, 1):
        print(f"{i}. {item}")
