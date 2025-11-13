import pandas as pd
import spacy
from nltk.corpus import stopwords
import nltk
import re

# Télécharger les stopwords anglais (une seule fois)
nltk.download('stopwords')

# Charger le modèle linguistique de spaCy
nlp = spacy.load("en_core_web_sm")

# Charger le dataset de films
films_df = pd.read_csv('../data/cleaned_movies.csv')

# 🧩 Combiner les colonnes textuelles utiles pour former un texte complet
# Tu peux ajuster selon les colonnes les plus informatives
films_df['text'] = (
    films_df['Title'].fillna('') + ' ' +
    films_df['Overview'].fillna('') + ' ' +
    films_df['Tagline'].fillna('') + ' ' +
    films_df['Genres'].fillna('') + ' ' +
    films_df['Keywords'].fillna('') + ' ' +
    films_df['Director'].fillna('') + ' ' +
    films_df['Cast'].fillna('')
)

# 🧹 Fonction de nettoyage et lemmatisation
def preprocess_text(text):
    text = text.lower()  # Minuscule
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprimer ponctuation et chiffres
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.text not in stopwords.words('english')
    ]
    return ' '.join(tokens)

# Application du prétraitement
print("🔄 Prétraitement des films en cours...")
films_df['clean_text'] = films_df['text'].apply(preprocess_text)
print("✅ Prétraitement terminé !")

# Sauvegarder le corpus nettoyé pour l’indexation
films_df[['Title', 'clean_text', 'Genres', 'Release_Date', 'Director', 'Vote_Average']].to_csv('../data/clean_corpus.csv', index=False)
print("💾 Corpus nettoyé enregistré dans data/clean_corpus.csv")
