import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Cargar datos
data = pd.read_csv("transporte_unsupervisado.csv")

# Codificar variables categóricas
le = LabelEncoder()
data['Trafico'] = le.fit_transform(data['Trafico'])
data['Clima'] = le.fit_transform(data['Clima'])
data['Dia'] = le.fit_transform(data['Dia'])

# Modelo K-Means
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(data)

# Mostrar resultados
print(data.head())

# Guardar resultados
data.to_csv("resultado_clustering.csv", index=False)