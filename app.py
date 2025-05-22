import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Récupérer le token depuis le fichier .env
SECRET_TOKEN = os.getenv('SECRET_TOKEN')

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle
model = joblib.load('stroke_model (1).pkl')

# Les 12 variables d’entrée attendues par le modèle
expected_features = [
    "Disease Free (Months)",
    "Person Neoplasm Status_WITH TUMOR",
    "Person Neoplasm Status_TUMOR FREE",
    "New Neoplasm Event Post Initial Therapy Indicator_NO",
    "Diagnosis Age",
    "New Neoplasm Event Post Initial Therapy Indicator_YES",
    "Patient Smoking History Category",
    "UICC TNM Tumor Stage Code_T2a",
    "Karnofsky Performance Score",
    "UICC TNM Tumor Stage Code_T2",
    "Prior Cancer Diagnosis Occurence_No",
    "UICC TNM Tumor Stage Code_T3b"
]

@app.route('/')
def home():
    return "✅ API de prédiction du risque d'AVC post-opératoire prête"

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer le header Authorization
    auth_header = request.headers.get('Authorization')

    # Vérifier le format "Bearer <token>"
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized: token manquant ou mal formaté'}), 401

    # Extraire le token réel
    token = auth_header.split(' ')[1]

    # Vérifier que le token correspond à celui attendu
    if token != SECRET_TOKEN:
        return jsonify({'error': 'Unauthorized: token invalide'}), 401

    try:
        data = request.get_json()

        # Vérifier les champs manquants
        missing = [f for f in expected_features if f not in data]
        if missing:
            return jsonify({'error': f'Champs manquants : {missing}'}), 400

        # Préparer les données
        input_data = [data[feature] for feature in expected_features]
        input_array = np.array(input_data).reshape(1, -1)

        # Prédiction
        proba = model.predict_proba(input_array)[0][1]
        prediction = int(proba >= 0.35)

        return jsonify({
            'prediction': prediction,
            'probability': round(proba, 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
