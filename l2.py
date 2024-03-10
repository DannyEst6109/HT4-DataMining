# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier

# Cargar datos
datos = pd.read_csv("./train.csv")

# Codificación one-hot de todas las variables categóricas
datos_encoded = pd.get_dummies(datos)

# 1. Utilizar los mismos conjuntos de entrenamiento y prueba que para regresión lineal
train, test = train_test_split(datos_encoded, test_size=0.3, random_state=123)

# 2. Imputar valores faltantes (NaN) en el conjunto de entrenamiento
imputer = SimpleImputer(strategy="mean")
train_imputed = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)

# 3. Ajustar el modelo de regresión lineal
fit_lm = LinearRegression().fit(train_imputed.drop(columns="SalePrice"), train_imputed["SalePrice"])

# 4. Imputar valores faltantes en el conjunto de prueba y predecir
test_imputed = pd.DataFrame(imputer.transform(test), columns=test.columns)
predictions_lm = fit_lm.predict(test_imputed.drop(columns="SalePrice"))

# 5. Evaluar el rendimiento del modelo
mse_lm = mean_squared_error(test_imputed["SalePrice"], predictions_lm)
print(f"MSE del modelo de regresión lineal: {mse_lm}")

# 6. Crear variable respuesta para clasificar las casas
datos["clasificacion"] = pd.cut(datos["SalePrice"], bins=[0, 170000, 290000, float('inf')], labels=["Economicas", "Intermedias", "Caras"])

# 7. Árbol de Clasificación
arbol_clasificacion = DecisionTreeClassifier()
arbol_clasificacion.fit(train.drop(columns=["SalePrice", "clasificacion"]), train["clasificacion"])

# 8. Evaluar eficiencia del modelo de clasificación con conjunto de prueba
predicciones_clasificacion = arbol_clasificacion.predict(test.drop(columns=["SalePrice", "clasificacion"]))
accuracy = accuracy_score(test["clasificacion"], predicciones_clasificacion)
print(f"Accuracy del árbol de clasificación: {accuracy}")

# 9. Matriz de confusión
cfm = confusion_matrix(test["clasificacion"], predicciones_clasificacion)
print("Matriz de Confusión:")
print(cfm)

# 10. Árbol de Clasificación con validación cruzada
arbol_cv = DecisionTreeClassifier()
scores = cross_val_score(arbol_cv, datos.drop(columns=["SalePrice", "clasificacion"]), datos["clasificacion"], cv=5)
print(f"Accuracy promedio con validación cruzada: {scores.mean()}")

# 11. Modelos adicionales cambiando la profundidad del árbol en validación cruzada
profundidades_cv = [5, 10, 15]
for profundidad in profundidades_cv:
    arbol_cv = DecisionTreeClassifier(max_depth=profundidad)
    scores = cross_val_score(arbol_cv, datos.drop(columns=["SalePrice", "clasificacion"]), datos["clasificacion"], cv=5)
    print(f"Accuracy promedio con validación cruzada para profundidad {profundidad}: {scores.mean()}")

# 12. Random Forest como algoritmo de predicción
random_forest = RandomForestClassifier()
random_forest.fit(train.drop(columns=["SalePrice", "clasificacion"]), train["clasificacion"])

# Comparar resultados entre árbol de clasificación y random forest
predicciones_rf = random_forest.predict(test.drop(columns=["SalePrice", "clasificacion"]))
accuracy_rf = accuracy_score(test["clasificacion"], predicciones_rf)
print(f"Accuracy del Random Forest: {accuracy_rf}")
