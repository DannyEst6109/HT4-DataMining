# Importar librerias
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import graphviz

# Cargar datos
datos = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv", dtype=str)

# Realizado anteriormente
# Inciso 4
set_entrenamiento = datos.sample(frac=0.7, random_state=42)
set_prueba = datos.loc[~datos.index.isin(set_entrenamiento.index)]

drop = ["LotFrontage", "Alley", "MasVnrType", "MasVnrArea", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "Electrical", "FireplaceQu", "GarageType", "GarageYrBlt",
        "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

set_entrenamiento = set_entrenamiento.drop(columns=drop)
set_prueba = set_prueba.drop(columns=drop)

# Convertir variables categóricas a dummy
set_entrenamiento = pd.get_dummies(set_entrenamiento, drop_first=True)
set_prueba = pd.get_dummies(set_prueba, drop_first=True)

# Elaborar árbol de regresión
X_train = set_entrenamiento.drop(columns="SalePrice")
y_train = set_entrenamiento["SalePrice"]
arbol_3 = DecisionTreeRegressor()
arbol_3.fit(X_train, y_train)


# Visualizar árbol
dot_data = tree.export_graphviz(arbol_3, out_file=None, feature_names=X_train.columns,
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Arbol_de_Regresion")

# 3. Predecir y analizar resultados
X_test = set_prueba.drop(columns="SalePrice")
y_test = set_prueba["SalePrice"]
predicciones = arbol_3.predict(X_test)

mse = mean_squared_error(y_test, predicciones)
print("MSE:", mse)

# 4. Hacer al menos 3 modelos más cambiando la profundidad del árbol
profundidades = [5, 10, 15, 3]
for profundidad in profundidades:
    arbol = DecisionTreeRegressor(max_depth=profundidad)
    arbol.fit(X_train, y_train)
    predicciones = arbol.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    print(f"MSE con profundidad {profundidad}: {mse}")
