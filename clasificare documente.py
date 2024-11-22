from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Generare date sintetice
data = {
    "text": [
        "Factură pentru serviciile prestate în luna octombrie, suma totală 1500 RON.",
        "Formular de înregistrare completat pentru pacientul Mihai Popescu.",
        "Chitanță pentru plata serviciilor de consultație medicală.",
        "Cerere de rambursare a cheltuielilor pentru tratamentele efectuate.",
        "Formular incomplet pentru înregistrarea pacientului.",
    ],
    "label": ["factură", "formular", "chitanță", "cerere", "formular"],
}

df = pd.DataFrame(data)

# Preprocesare și împărțirea datelor
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformare text -> vectori folosind TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Antrenarea clasificatorului Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_vec, y_train)

# Predicții și evaluare
y_pred = clf.predict(X_test_vec)

print("Acuratețea:", accuracy_score(y_test, y_pred))
print("Raport clasificare:\n", classification_report(y_test, y_pred))
