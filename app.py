from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Chargement du modèle
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Modèle chargé avec succès")
except FileNotFoundError:
    print("❌ Fichier model.pkl introuvable. Assurez-vous de l'avoir créé.")
    model = None

# Template HTML avec CSS et JS intégré
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification de Fruits 🍎🍊</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 500px;
            width: 100%;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
            font-size: 0.95em;
        }
        
        select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        select:hover {
            border-color: #667eea;
        }
        
        select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
            font-weight: 600;
            display: none;
            animation: slideIn 0.5s;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result.apple {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
        }
        
        .result.orange {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .result-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .info-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 0.85em;
            color: #666;
        }
        
        .info-box strong {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍎 Classification de Fruits 🍊</h1>
        <p class="subtitle">Prédiction basée sur l'IA</p>
        
        <form id="fruitForm">
            <div class="form-group">
                <label for="couleur">Couleur du fruit</label>
                <select id="couleur" name="couleur" required>
                    <option value="">-- Choisissez --</option>
                    <option value="0">🟢 Vert</option>
                    <option value="1">🔴 Rouge</option>
                    <option value="2">🟠 Orange</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="texture">Texture du fruit</label>
                <select id="texture" name="texture" required>
                    <option value="">-- Choisissez --</option>
                    <option value="0">✨ Lisse</option>
                    <option value="1">🌰 Rugueux</option>
                </select>
            </div>
            
            <button type="submit">🔍 Classifier le fruit</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyse en cours...</p>
        </div>
        
        <div class="result" id="result">
            <div class="result-icon" id="resultIcon"></div>
            <div id="resultText"></div>
        </div>
        
        <div class="info-box">
            <strong>Comment ça marche ?</strong><br>
            Ce modèle utilise un arbre de décision entraîné sur des caractéristiques simples 
            (couleur et texture) pour prédire si le fruit est une pomme ou une orange.
        </div>
    </div>

    <script>
        const form = document.getElementById('fruitForm');
        const result = document.getElementById('result');
        const resultIcon = document.getElementById('resultIcon');
        const resultText = document.getElementById('resultText');
        const loading = document.getElementById('loading');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const couleur = document.getElementById('couleur').value;
            const texture = document.getElementById('texture').value;
            
            if (!couleur || !texture) {
                alert('Veuillez remplir tous les champs');
                return;
            }
            
            // Afficher le loading
            result.style.display = 'none';
            loading.style.display = 'block';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        couleur: parseInt(couleur),
                        texture: parseInt(texture)
                    })
                });
                
                const data = await response.json();
                
                // Masquer le loading
                loading.style.display = 'none';
                
                if (data.success) {
                    // Afficher le résultat
                    result.style.display = 'block';
                    
                    if (data.prediction === 'pomme') {
                        result.className = 'result apple';
                        resultIcon.textContent = '🍎';
                        resultText.innerHTML = `C'est une <strong>POMME</strong> !<br><small>Probabilité: ${data.probability}%</small>`;
                    } else {
                        result.className = 'result orange';
                        resultIcon.textContent = '🍊';
                        resultText.innerHTML = `C'est une <strong>ORANGE</strong> !<br><small>Probabilité: ${data.probability}%</small>`;
                    }
                } else {
                    alert('Erreur: ' + data.error);
                }
            } catch (error) {
                loading.style.display = 'none';
                alert('Erreur de connexion au serveur: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

# ---- ROUTES ----

@app.route('/')
def index():
    """Page d'accueil avec le formulaire"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint pour la prédiction"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Modèle non chargé'
        }), 500
    
    try:
        # Récupération des données JSON
        data = request.get_json()
        
        if not data or 'couleur' not in data or 'texture' not in data:
            return jsonify({
                'success': False,
                'error': 'Données manquantes (couleur et texture requis)'
            }), 400
        
        couleur = int(data['couleur'])
        texture = int(data['texture'])
        
        # Validation des valeurs
        if couleur not in [0, 1, 2] or texture not in [0, 1]:
            return jsonify({
                'success': False,
                'error': 'Valeurs invalides'
            }), 400
        
        # Prédiction
        features = np.array([[couleur, texture]])
        prediction = model.predict(features)[0]
        
        # Probabilités (si disponible)
        try:
            probabilities = model.predict_proba(features)[0]
            probability = round(max(probabilities) * 100, 2)
        except:
            probability = 100
        
        # Conversion en nom de fruit
        fruit_name = 'pomme' if prediction == 0 else 'orange'
        
        return jsonify({
            'success': True,
            'prediction': fruit_name,
            'prediction_code': int(prediction),
            'probability': probability,
            'input': {
                'couleur': couleur,
                'texture': texture
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """Informations sur l'API"""
    return jsonify({
        'name': 'Fruit Classifier API',
        'version': '1.0',
        'model': 'DecisionTreeClassifier',
        'features': {
            'couleur': {
                'type': 'categorical',
                'values': {0: 'vert', 1: 'rouge', 2: 'orange'}
            },
            'texture': {
                'type': 'categorical',
                'values': {0: 'lisse', 1: 'rugueux'}
            }
        },
        'output': {
            'type': 'categorical',
            'values': {0: 'pomme', 1: 'orange'}
        }
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Démarrage de l'application Flask")
    print("="*50)
    print("📍 URL: http://127.0.0.1:5000")
    print("📚 API Info: http://127.0.0.1:5000/api/info")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)