import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def run_decision_tree_analysis(visualize=False):

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
      # Entrenar modelo de regresión con la profundidad actual
      modelo = DecisionTreeRegressor(max_depth=profundidad, random_state=42)
      modelo.fit(X_train, y_train)
      
      # Evaluar el modelo
      predicciones = modelo.predict(X_test)
      mse = mean_squared_error(y_test, predicciones)
      
      # Guardar los resultados (profundidad, MSE)
      resultados.append((profundidad, mse))

  # Imprimir los resultados y encontrar el modelo con menor MSE
  mejor_resultado = min(resultados, key=lambda item: item[1])
  mejor_profundidad, mejor_mse = mejor_resultado
  print(f"\n\nLa mejor profundidad es {mejor_profundidad} con un MSE de {mejor_mse}")

  # La mejor profundidad es aquella con el menor MSE
  # Reentrenar el modelo con la mejor profundidad encontrada
  mejor_modelo = DecisionTreeRegressor(max_depth=mejor_profundidad, random_state=42)
  mejor_modelo.fit(X_train, y_train)

  if visualize:
    # Visualizar el mejor árbol de decisión
    # Modifica el valor de max_depth aquí si deseas ver más niveles del árbol
    profundidad_visualizacion = 2
    plt.figure(figsize=(30, 20))
    plot_tree(mejor_modelo, filled=True, feature_names=X.columns, fontsize=10, max_depth=profundidad_visualizacion)
    plt.show()

  return mejor_mse, mejor_modelo

# Ejecutar análisis de árbol de decisión
if __name__ == "__main__":
  run_decision_tree_analysis(visualize=True)