import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Cargar datos
data = pd.read_csv("transporte.csv")

# Codificar variables
le = LabelEncoder()
data['Trafico'] = le.fit_transform(data['Trafico'])
data['Clima'] = le.fit_transform(data['Clima'])
data['Dia'] = le.fit_transform(data['Dia'])
data['Retraso'] = le.fit_transform(data['Retraso'])

# Variables
X = data[['Hora', 'Trafico', 'Clima', 'Pasajeros', 'Dia']]
y = data['Retraso']

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))
