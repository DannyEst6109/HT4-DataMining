# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
datos = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv", dtype=str)

# Convertir columnas no numéricas a numéricas usando one-hot encoding
datos = pd.get_dummies(datos)
test = pd.get_dummies(test)

# Dividir los datos
set_entrenamiento, set_prueba = train_test_split(datos, test_size=0.3, random_state=123)

# Columnas a eliminar
drop_columns = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "Electrical", "FireplaceQu", "GarageType", "GarageFinish",
                "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

# Eliminar columnas solo si existen en el conjunto de entrenamiento
set_entrenamiento = set_entrenamiento.drop(columns=[col for col in drop_columns if col in set_entrenamiento.columns])
set_prueba = set_prueba.drop(columns=[col for col in drop_columns if col in set_prueba.columns])

# Árbol de Regresión
arbol_regresion = DecisionTreeRegressor()
arbol_regresion.fit(set_entrenamiento.drop(columns="SalePrice"), set_entrenamiento["SalePrice"])

# Visualizar el árbol de regresión
plt.figure(figsize=(12, 8))
plt.title("Árbol de Regresión")
_ = plot_tree(arbol_regresion, filled=True, feature_names=set_entrenamiento.drop(columns="SalePrice").columns)

# Predicciones y evaluación del árbol de regresión
predicciones = arbol_regresion.predict(set_prueba.drop(columns="SalePrice"))
mse = mean_squared_error(set_prueba["SalePrice"], predicciones)
print(f"MSE: {mse}")

# Modelos adicionales cambiando la profundidad del árbol
profundidades = [5, 10, 15, 3]
for profundidad in profundidades:
    arbol = DecisionTreeRegressor(max_depth=profundidad)
    arbol.fit(set_entrenamiento.drop(columns="SalePrice"), set_entrenamiento["SalePrice"])
    predicciones = arbol.predict(set_prueba.drop(columns="SalePrice"))
    mse = mean_squared_error(set_prueba["SalePrice"], predicciones)
    print(f"MSE para profundidad {profundidad}: {mse}")

# Comparación con modelo de regresión lineal
train, test = train_test_split(datos, test_size=0.3, random_state=123)
fit_lm = LinearRegression().fit(train.drop(columns="SalePrice"), train["SalePrice"])
predictions_lm = fit_lm.predict(test.drop(columns="SalePrice"))
mse_lm = mean_squared_error(test["SalePrice"], predictions_lm)
print(f"MSE del modelo de regresión lineal: {mse_lm}")

# Crear variable respuesta
datos["clasificacion"] = pd.cut(datos["SalePrice"], bins=[0, 170000, 290000, float('inf')], labels=["Economicas", "Intermedias", "Caras"])

# Árbol de Clasificación
arbol_clasificacion = DecisionTreeClassifier()
arbol_clasificacion.fit(set_entrenamiento.drop(columns=["SalePrice", "clasificacion"]), set_entrenamiento["clasificacion"])

# Visualizar el árbol de clasificación
plt.figure(figsize=(12, 8))
plt.title("Árbol de Clasificación")
_ = plot_tree(arbol_clasificacion, filled=True, feature_names=set_entrenamiento.drop(columns=["SalePrice", "clasificacion"]).columns, class_names=arbol_clasificacion.classes_)

# Evaluar eficiencia del modelo de clasificación
predicciones_clasificacion = arbol_clasificacion.predict(set_prueba.drop(columns=["SalePrice", "clasificacion"]))
accuracy = accuracy_score(set_prueba["clasificacion"], predicciones_clasificacion)
print(f"Accuracy del modelo de clasificación: {accuracy}")

# Matriz de confusión
cfm = confusion_matrix(set_prueba["clasificacion"], predicciones_clasificacion)
print("Matriz de Confusión:")
print(cfm)
