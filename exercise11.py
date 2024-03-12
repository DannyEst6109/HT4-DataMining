import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Carga de datos
df = pd.read_csv('train.csv')

# Aplicar codificación one-hot a variables categóricas
df = pd.get_dummies(df)

# Separar las características (X) y la variable objetivo (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista para almacenar los resultados de cada profundidad
resultados = []

# Profundidades a probar
profundidades = [1, 2, 5, 10, None]

for profundidad in profundidades:
    # Entrenar modelo con la profundidad actual
    modelo = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Evaluar el modelo
    predicciones = modelo.predict(X_test)
    score = accuracy_score(y_test, predicciones)
    
    # Guardar los resultados
    resultados.append((profundidad, score))

# Imprimir los resultados
for profundidad, score in resultados:
    print(f"Profundidad: {profundidad}, Precisión: {score}")

# Encontrar la mejor profundidad y reentrenar el modelo
mejor_profundidad = max(resultados, key=lambda item: item[1])[0]
mejor_modelo = DecisionTreeClassifier(max_depth=mejor_profundidad, random_state=42)
mejor_modelo.fit(X_train, y_train)

# Visualizar solo las primeras capas del mejor árbol de decisión
plt.figure(figsize=(30, 20))
plot_tree(mejor_modelo, filled=True, feature_names=X.columns, fontsize=10, max_depth=2)
plt.show()