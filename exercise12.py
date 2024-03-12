import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
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

# Configurar los parámetros y el modelo para la búsqueda en cuadrícula
parametros_rf = {
    'n_estimators': [100, 200, 300],  # Número de árboles
    'max_depth': [None, 10, 20],       # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10]    # Mínimo número de muestras requeridas para dividir un nodo
}

rf = RandomForestRegressor(random_state=42)

# Realizar la búsqueda en cuadrícula
grid_search_rf = GridSearchCV(estimator=rf, param_grid=parametros_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# Obtener el mejor modelo
mejor_rf = grid_search_rf.best_estimator_

# Evaluar el mejor modelo usando el conjunto de prueba
predicciones_rf = mejor_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, predicciones_rf)

print(f"El mejor Random Forest tiene un MSE de: {mse_rf}")

# Visualizar la importancia de las características
importancias = mejor_rf.feature_importances_
indices_importantes = np.argsort(importancias)[::-1]
n_caracteristicas = 10  # Mostrar las 10 características más importantes
nombres_caracteristicas = [X.columns[i] for i in indices_importantes[:n_caracteristicas]]

plt.figure(figsize=(12, 6))
plt.title('Importancia de las Características Top 10')
plt.bar(range(n_caracteristicas), importancias[indices_importantes[:n_caracteristicas]], color='skyblue', align='center')
plt.xticks(range(n_caracteristicas), nombres_caracteristicas, rotation=45, ha='right')
plt.xlim([-1, n_caracteristicas])
plt.tight_layout()
plt.show()