import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Assure-toi que NLTK est prêt
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Charger le modèle et le vectoriseur
try:
    model = joblib.load('sentiment_analysis1_model.pkl')  # Assure-toi que le nom du fichier est correct
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Modèle et vectoriseur chargés avec succès.")
except FileNotFoundError as e:
    print(f"Erreur : {e}. Vérifie que le fichier existe dans le bon répertoire.")
    exit()

# Fonction de nettoyage (préprocessus)
def preprocess_review(review):
    review = re.sub(r'[^\w\s]', '', review)  # Enlève les caractères spéciaux
    tokens = word_tokenize(review.lower())   # Tokenisation
    tokens = [word for word in tokens if word not in stop_words]  # Suppression des stopwords
    return ' '.join(tokens)

# Fonction de prédiction
def predict_sentiment(review):
    cleaned_review = preprocess_review(review)  # Nettoyage de l'avis
    review_vector = vectorizer.transform([cleaned_review])  # Vectorisation
    prediction = model.predict(review_vector)  # Prédiction
    if prediction[0] == 1:
        return "Négatif"
    else:
        return "Positif"

# Exemple d'avis à prédire
new_review = "I hate this product, don't buy it !!"
print(f"L'avis est : {predict_sentiment(new_review)}")