#  CineFinder  - Moteur de Recherche de Films

Une application web intelligente pour découvrir des films selon vos envies, utilisant deux approches de recherche complémentaires : **BM25 (Recherche par Mots-clés)** et **BERT Sémantique (Compréhension du Sens)**.

---

##  Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Deux Moteurs de Recherche](#deux-moteurs-de-recherche)
- [Installation](#installation)
- [Démarrage Rapide](#démarrage-rapide)

---

##  Vue d'ensemble

**CineFinder** combine deux approches de recherche puissantes pour offrir une expérience utilisateur optimale :

1. **Recherche Traditionnelle (BM25)** : Basée sur les mots-clés exacts
2. **Recherche Sémantique (BERT)** : Basée sur la compréhension du sens

Les utilisateurs peuvent choisir le moteur qui correspond le mieux à leur besoin en temps réel.

---

##  Deux Moteurs de Recherche

### 1️ BM25 + Classification Automatique

#### Comment ça fonctionne

Le moteur BM25 utilise une approche **traditionnelle et déterministe** basée sur :

- **Extraction de tokens** : Analyse du texte avec spaCy (lemmatisation, suppression des mots vides)
- **Indexation inversée** : Création d'un index pour chaque champ (Titre, Réalisateur, Genre, etc.)
- **Classification intelligente** : Reconnaissance automatique du type de requête (année, genre, réalisateur, titre)
- **Scoring BM25** : Calcul d'un score de pertinence pour chaque document

#### Cas d'usage idéaux

- Recherches **précises** : "Nolan 2010", "Tarantino western"
- Recherches par **genre** ou **année** : "Action 2020"
- Recherches par **réalisateur** ou **acteur**
- Requêtes **courtes et directives**

#### Avantages

✅ Résultats **prévisibles et consistants**  
✅ Aucune dépendance à un modèle d'IA externe  
✅ Index **pré-calculé et sauvegardé** (démarrage ultra-rapide)  
✅ Comprend **les années, genres, réalisateurs** automatiquement  
✅ Permet **les filtres avancés** sans surcoût

---

### 2️ Recherche Sémantique BERT

#### Comment ça fonctionne

Le moteur sémantique utilise **BERT** (Sentence-Transformers) pour comprendre le **sens** des phrases :

- **Embedding de requête** : Conversion de la requête en vecteur numérique représentant le sens
- **Embeddings pré-calculés** : Tous les films sont convertis en embeddings stockés dans un fichier
- **Similarité cosinus** : Comparaison du vecteur de requête avec les embeddings des films
- **Scoring par pertinence sémantique** : Classement par similarité de sens

#### Cas d'usage idéaux

- Recherches **descriptives** : "un film sur une famille qui s'infiltre dans la richesse"
- Recherches **thématiques** : "films sur le temps et l'espace"
- Recherches **par ambiance** : "un film sombre et psychologique"
- Requêtes **longues et narratives**
- Comprendre l'**intention derrière la recherche**

#### Avantages

✅ Comprend le **contexte et le sens**  
✅ Fonctionne avec **des descriptions naturelles**  
✅ Tolère les **fautes de frappe et variantes**  
✅ Excellent pour les **requêtes complexes**  
✅ Score de similarité **transparent** (0-1)

---
## 📸 Exemples d'Utilisation

### Exemple 1 : Recherche par Mots-clés (BM25)
**Requête** : "batman"

![Résultats BM25 - Batman](https://github.com/user-attachments/assets/bfb14f25-bc51-41c4-83c8-7b297261e7f7)

**Résultats** : 4 films trouvés avec le moteur BM25
- Batman (1989)
- Batman Returns (1992)
- Batman Begins (2005)

**Avantage BM25** : Reconnaissance immédiate du mot-clé "Batman", résultats précis et directs

---

### Exemple 2 : Recherche Sémantique (BERT)
**Requête** : "zombie apocalypse"

![Résultats BERT - Zombie Apocalypse](https://github.com/user-attachments/assets/175ecdfe-d7bc-4634-8574-40f338623715)


**Résultats** : 7 films trouvés avec le moteur BERT
- Pontypool (54% Match)
- Resident Evil: Apocalypse (53% Match)
- Land of the Dead (53% Match)
- Day of the Dead (52% Match)

**Avantage BERT** : Compréhension du concept "apocalypse zombie", résultats diversifiés et pertinents par thème

---


##  Comparaison Rapide

| Aspect | BM25 | BERT Sémantique |
|--------|------|-----------------|
| **Type de recherche** | Mots-clés exacts | Sens et contexte |
| **Temps de démarrage** | ⚡ Très rapide | 🕐 Modéré (modèle à charger) |
| **Qualité requête simple** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Qualité requête complexe** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Compréhension genre/année** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Compréhension narrative** | ⭐ | ⭐⭐⭐⭐⭐ |
| **Index sauvegardé** | ✅ Oui | ✅ Oui |
| **Filtres avancés** | ✅ Supportés | ⚠️ Limités |

---

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip ou conda

### Étapes

#### 1. Cloner le repository

```bash
git clone https://github.com/ily1s/CineFinder
cd CineFinder
```

#### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

#### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```
#### 4. Télécharger le modèle spaCy

```bash
python -m spacy download en_core_web_sm
```



---

##  Démarrage Rapide

### Lancer l'interface Gradio

```bash
    streamlit run app.py
```

Ouvrez votre navigateur : `http://localhost:8502`

### Interface Utilisateur

1. **Barre de recherche** : Décrivez le film recherché
2. **Sélection du moteur** : BM25 ou BERT
3. **Filtres avancés** (optionnel) : Genre, année, note, durée, réalisateur, acteur
4. **Résultats** : Les résultats des films avec leurs détails   

---


### Flux de Données

```
DÉMARRAGE
    ↓
├─→ Charger metadonnées (CSV)
├─→ Charger embeddings BERT (pickle)
└─→ Charger index BM25 (JSON)
    ↓
RECHERCHE
    ↓
├─→ BM25 : Indexation inversée → Scoring BM25 → Top-N
└─→ BERT : Embedding requête → Similarité cosinus → Top-N
    ↓
FILTRAGE
    ↓
├─→ Genre, Année, Note, Durée
├─→ Réalisateur, Acteur
    ↓
AFFICHAGE
    ↓
└─→ Grille responsive avec modal détails
```

---