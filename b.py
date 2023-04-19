# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Cargar los datos
df = pd.read_csv('games.csv', sep=',')

# Cambiar nombres de las columnas al español
df.columns = ['ID', 'Título', 'Fecha de Lanzamiento', 'Equipo', 'Calificación', 'Veces Listado', 'Número de Reseñas', 'Géneros', 'Resumen', 'Reseñas', 'Veces Jugado', 'Jugando', 'Pendientes', 'Lista de Deseos']

# Preparar los datos
def multiply_k(x):
    if 'K' in str(x):
        return float(x.replace('K', '')) * 1000
    else:
        return float(x)

# Convierte las columnas Calificación y Veces Jugado a números
df['Veces Jugado'] = df['Veces Jugado'].apply(lambda x: multiply_k(x))
df['Número de Reseñas'] = df['Número de Reseñas'].apply(lambda x: multiply_k(x))

df['Veces Jugado'] = (df['Veces Jugado'] - df['Veces Jugado'].min()) / (df['Veces Jugado'].max() - df['Veces Jugado'].min()) * 100
df['Número de Reseñas'] = (df['Número de Reseñas'] - df['Número de Reseñas'].min()) / (df['Número de Reseñas'].max() - df['Número de Reseñas'].min()) * 100

# Seleccionar dos variables para la regresión lineal
x = df['Veces Jugado'].values.reshape(-1, 1)  # Variables independientes
y = df['Número de Reseñas'].values.reshape(-1, 1)  # Variable dependiente

# Dividir los datos en conjunto de entrenamiento y de prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=37)
"""
test_size: la proporción del conjunto de datos que se utilizará para la prueba. Si se establece en 0.2, el 20% 
del conjunto de datos se utilizará para la prueba y el 80% se utilizará para el entrenamiento.
random_state: se utiliza para inicializar el generador de números aleatorios que realiza la división aleatoria.
 Si se establece en un número entero, el resultado de la división aleatoria será el mismo cada vez que se ejecute el código.
"""

# Crear el modelo de regresión lineal
reg_model = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
reg_model.fit(x_train, y_train)

# Coeficientes de la regresión lineal
a = reg_model.intercept_[0]
b = reg_model.coef_[0][0]

# Predecir los valores de y utilizando los datos de prueba
y_pred = reg_model.predict(x_test)

# Error cuadrático medio (MSE) utilizando los datos de prueba
mse = mean_squared_error(y_test, y_pred)

# Coeficiente de determinación R^2 utilizando los datos de prueba
r2 = r2_score(y_test, y_pred)

print("Coeficientes de la regresión lineal: a = {:.2f}, b = {:.2f}".format(a, b))
print("Error cuadrático medio (MSE): {:.2f}".format(mse))
print("Coeficiente de determinación R^2: {:.2f}".format(r2))


# Crear la figura y los ejes
fig, ax = plt.subplots()

# Graficar los puntos de datos de prueba
ax.scatter(x_test, y_test, color='blue')

# Graficar la regresión lineal utilizando los datos de prueba
ax.plot(x_test, y_pred, color='red')

# Configurar los ejes y las etiquetas
ax.set_xlabel('Veces Jugado')
ax.set_ylabel('Número de Reseñas')
ax.set_title('Regresión Lineal')

# Mostrar la gráfica
plt.show()

"""
CONCLUSION:
Los coeficientes de la regresión lineal obtenidos para los datos de prueba son a = 4.30 y b = 0.71.
Esto significa que por cada vez que se juega el juego, se espera que el número de reseñas aumente en 4.30 unidades,
y que incluso cuando el juego no se juega en absoluto, se espera que tenga 0.71 reseñas.
El error cuadrático medio (MSE) para los datos de prueba es de 89.61.
Esto indica que la línea recta ajustada no se ajusta muy bien a los datos de prueba 
y que hay una gran cantidad de variabilidad en el número de reseñas que se reciben para un número determinado de veces que se juega el juego.
El coeficiente de determinación R^2 para los datos de prueba es de 0.70+,
lo que indica que la línea recta ajustada explica una cantidad moderada de la variabilidad en los datos de prueba.

Comparando los resultados obtenidos con los datos de prueba con los obtenidos con los datos anteriores,
se puede observar que los coeficientes de la regresión lineal son ligeramente diferentes,
con un valor de a ligeramente mayor y un valor de b ligeramente menor en los datos de prueba. 
Además, el MSE para los datos de prueba es ligeramente mayor que el obtenido anteriormente,
lo que indica que la línea recta ajustada no se ajusta tan bien a los datos de prueba como lo hizo a los datos anteriores.
Sin embargo, el coeficiente de determinación R^2 es ligeramente mayor para los datos de prueba,
lo que indica que la línea recta ajustada explica una mayor cantidad de la variabilidad en los datos de prueba que en los datos anteriores.
"""