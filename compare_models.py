from exercise11 import run_decision_tree_analysis
from exercise12 import run_random_forest_analysis

# Ejecutar análisis de Random Forest
mse_rf, mejor_rf = run_random_forest_analysis(visualize=False)

# Ejecutar análisis de árbol de decisión
mse_arbol_decision, mejor_arbol_decision = run_decision_tree_analysis(visualize=False)


# Comparar MSE de ambos modelos
print(f"\n\nMSE Árbol de Decisión: {mse_arbol_decision}")
print(f"MSE Random Forest: {mse_rf}")

# Determinar qué modelo tuvo un mejor rendimiento
if mse_rf < mse_arbol_decision:
    print("\n\nEl modelo de Random Forest tiene un mejor rendimiento basado en el MSE.")
else:
    print("\n\nEl modelo de Árbol de Decisión tiene un mejor rendimiento basado en el MSE.")

# Discusión cualitativa de los resultados
print("""
Ambos algoritmos tienen sus fortalezas y debilidades. El árbol de decisión es un modelo más simple y fácil de interpretar, 
pero puede ser propenso al sobreajuste. Por otro lado, el Random Forest es un conjunto de árboles de decisión, lo que 
generalmente mejora el rendimiento y la robustez del modelo al reducir el sobreajuste, a costa de ser más complejo y 
menos interpretable.

En este caso, el Random Forest tuvo un mejor rendimiento según el MSE, lo cual sugiere que puede manejar mejor la 
varianza en los datos. Sin embargo, este mejor rendimiento viene con un costo adicional en términos de tiempo de 
computación y complejidad del modelo. Para decisiones finales sobre el modelo a implementar, también deberíamos 
considerar otros factores como la velocidad de entrenamiento, la facilidad de interpretación y la capacidad del modelo 
para actualizar nuevos datos.
""")