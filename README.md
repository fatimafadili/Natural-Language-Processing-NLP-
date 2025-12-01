📝 NLP - Natural Language Processing
🎯 Introduction au Natural Language Preprocessing
Ce projet éducatif couvre toutes les étapes fondamentales du Natural Language Processing (NLP) en utilisant TensorFlow et Keras. L'objectif est de comprendre comment transformer du texte brut en données exploitables par des modèles de machine learning.

📁 Structure du Projet
text
NLP_PROJECT/
├── 📄 README.md                          # Ce fichier
├── 📁 notebooks/
│   ├── 📘 01_basic_tokenization.ipynb    # Tokenization basique
│   ├── 📘 02_advanced_preprocessing.ipynb # Préprocessing avancé
│   ├── 📘 03_embeddings.ipynb            # Word embeddings
│   └── 📘 04_text_classification.ipynb   # Classification de texte
├── 📁 data/
│   ├── 📊 sarcasm.json                   # Dataset sarcasme
│   ├── 📊 imdb_reviews.csv               # Reviews IMDB
│   └── 📊 custom_data/                   # Données personnalisées
├── 📁 src/
│   ├── 🐍 preprocessor.py                # Classes de préprocessing
│   ├── 🐍 models.py                      # Architectures de modèles
│   └── 🐍 utils.py                       # Fonctions utilitaires
└── requirements.txt                      # Dépendances Python
🚀 Installation Rapide
bash
# 1. Cloner le projet
git clone https://github.com/votre-username/nlp-project.git
cd nlp-project

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur Mac/Linux :
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt
requirements.txt :

text
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
nltk>=3.7
scikit-learn>=1.0.0
📚 Concepts Clés Couverts
🔤 1. Tokenization Basique
python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
]

# Création du tokenizer
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

# Résultat
word_index = tokenizer.word_index
print(word_index)
# {'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}
🔠 2. Gestion des Mots Inconnus (OOV)
python
# Tokenizer avec token pour mots hors vocabulaire
tokenizer_oov = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer_oov.fit_on_texts(sentences)

# Test avec nouveau texte
new_sentence = ["I love my phone"]
new_sequence = tokenizer_oov.texts_to_sequences(new_sentence)
print(new_sequence)  # [[3, 1, 2, 1]]  # 'phone' -> <OOV>
📏 3. Padding des Séquences
python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Conversion en séquences
sequences = tokenizer.texts_to_sequences(sentences)

# Application du padding
padded = pad_sequences(
    sequences,
    maxlen=10,
    padding='post',
    truncating='post'
)
print(padded)
🧹 4. Nettoyage de Texte Avancé
python
import re
import nltk
from nltk.corpus import stopwords

def clean_text(text):
    """Nettoie le texte en plusieurs étapes"""
    # 1. Minuscules
    text = text.lower()
    
    # 2. Suppression URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Suppression ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Suppression stopwords (optionnel)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text
🧠 5. Word Embeddings
python
from tensorflow.keras.layers import Embedding

# Création d'une couche d'embedding
embedding_layer = Embedding(
    input_dim=10000,     # Taille du vocabulaire
    output_dim=128,      # Dimension des embeddings
    input_length=100     # Longueur des séquences
)

# Utilisation avec un modèle
model = tf.keras.Sequential([
    embedding_layer,
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
🎮 Utilisation Pas-à-Pas
Étape 1 : Chargement des Données
python
import json
import pandas as pd

# Chargement du dataset sarcasme
def load_sarcasm_data(filepath):
    with open(filepath, 'r') as f:
        datastore = json.load(f)
    
    sentences = []
    labels = []
    
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
    
    return sentences, labels

sentences, labels = load_sarcasm_data('data/sarcasm.json')
Étape 2 : Préprocessing Complet
python
class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        self.max_sequence_length = max_sequence_length
    
    def fit_transform(self, texts):
        # 1. Nettoyage
        cleaned_texts = [clean_text(text) for text in texts]
        
        # 2. Tokenization
        self.tokenizer.fit_on_texts(cleaned_texts)
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # 3. Padding
        padded = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        return padded
    
    def transform(self, texts):
        # Transformation sur de nouveaux textes
        cleaned_texts = [clean_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        return padded
Étape 3 : Création du Modèle
python
def create_text_classification_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        # Couche d'embedding
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),
        
        # Pooling global
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Couches denses
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Couche de sortie
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
Étape 4 : Entraînement
python
# Initialisation du preprocessor
preprocessor = TextPreprocessor(max_vocab_size=10000, max_sequence_length=100)

# Préparation des données
X_train = preprocessor.fit_transform(sentences)
y_train = np.array(labels)

# Création du modèle
model = create_text_classification_model(
    vocab_size=preprocessor.tokenizer.num_words,
    embedding_dim=128,
    max_length=100
)

# Entraînement
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
Étape 5 : Évaluation
python
# Visualisation des résultats
def plot_training_history(history):
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Courbe de perte
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.legend()
    
    # Courbe de précision
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.show()

plot_training_history(history)
📊 Applications Pratiques
1. Classification de Sentiments
python
# Modèle LSTM pour analyse de sentiments
sentiment_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
2. Détection de Sarcasme
python
# Modèle convolutif pour détection de sarcasme
sarcasm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
3. Génération de Texte
python
# Modèle pour génération de texte
text_generation_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 256),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])
📈 Métriques d'Évaluation
Métrique	Formule	Usage
Accuracy	(TP+TN)/(TP+TN+FP+FN)	Performance globale
Precision	TP/(TP+FP)	Qualité des prédictions positives
Recall	TP/(TP+FN)	Capacité à trouver tous les positifs
F1-Score	2(PrecisionRecall)/(Precision+Recall)	Moyenne harmonique
🔧 API Référence
Tokenizer
python
Tokenizer(
    num_words=None,          # Nombre max de mots dans le vocabulaire
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,              # Convertir en minuscules
    split=' ',               # Séparateur
    char_level=False,        # Tokenization par caractère
    oov_token=None,          # Token pour mots inconnus
    document_count=0
)
pad_sequences
python
pad_sequences(
    sequences,
    maxlen=None,            # Longueur maximale
    dtype='int32',
    padding='pre',          # 'pre' ou 'post'
    truncating='pre',       # 'pre' ou 'post'
    value=0.0               # Valeur de padding
)
🐛 Débogage Common
Problème 1 : Vocabulaire trop petit
python
# Solution : Augmenter num_words
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
Problème 2 : Séquences de longueurs différentes
python
# Solution : Padding uniforme
padded = pad_sequences(sequences, maxlen=100, padding='post')
Problème 3 : Overfitting
python
# Solution : Ajouter de la régularisation
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
📚 Ressources Supplémentaires
Documentation Officielle
TensorFlow Text Processing

Keras Preprocessing

NLTK Documentation

Datasets
Sarcasm Dataset - 26,000 headlines avec labels sarcastiques

IMDB Reviews - 50,000 reviews de films

20 Newsgroups - 20,000 documents texte

Reuters News - 11,228 articles de presse

Livres Recommandés
"Speech and Language Processing" - Jurafsky & Martin

"Natural Language Processing with Python" - Bird, Klein & Loper

"Deep Learning for NLP" - Goldberg
