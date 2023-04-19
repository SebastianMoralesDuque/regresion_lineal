import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lee el archivo CSV
df = pd.read_csv('games.csv', delimiter=',')

# Cambia los nombres de las columnas al español
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

# Convierte los datos relevantes en arrays de NumPy
x = df['Veces Jugado'].values
y = df['Número de Reseñas'].values

# Calcula los coeficientes de la regresión lineal por el método de mínimos cuadrados
N = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_xx = np.sum(x**2)
b = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x**2)
a = np.mean(y) - b * np.mean(x)

# Calcula el error cuadrático medio (MSE)
y_pred = a + b * x
mse = np.mean((y - y_pred)**2)

# Calcula el coeficiente de determinación R^2
ss_tot = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_pred)**2)
r2 = 1 - ss_res / ss_tot

# Grafica los datos y la regresión lineal
# Crear la figura y los ejes
fig, ax = plt.subplots()

# Imprime los resultados
print(f"Coeficientes de la regresión lineal: a = {a:.2f}, b = {b:.2f}")
print(f"Error cuadrático medio (MSE): {mse:.2f}")
print(f"Coeficiente de determinación R^2: {r2:.2f}")

plt.scatter(x, y, alpha=0.5)
plt.plot(x, y_pred, color='red')
plt.title('Regresión Lineal por Mínimos Cuadrados')
plt.xlabel('Veces Jugado')
plt.ylabel('Calificación')
plt.show()

"""
CONCLUSION:

En este caso, la ecuación de la línea recta es y = 4.02x + 0.73. Esto significa que por cada vez que se juega el juego, 
se espera que el número de reseñas aumente en 4.02 unidades, y que incluso cuando el juego no se juega en absoluto, se espera que tenga 0.73 reseñas.

El error cuadrático medio (MSE) es una medida de qué tan bien se ajusta la línea recta a los datos. 
Un MSE bajo indica que la línea recta se ajusta bien a los datos, mientras que un MSE alto indica que la línea recta no se ajusta bien a los datos.
En este caso, el MSE es de 84.83, lo que indica que la línea recta no se ajusta muy bien a los datos 
y que hay una gran cantidad de variabilidad en el número de reseñas que se reciben para un número determinado de veces que se juega el juego.

El coeficiente de determinación R^2 es una medida de cuánto de la variabilidad en los datos es explicada por la línea recta ajustada. 
Un R^2 alto indica que la línea recta explica bien la variabilidad en los datos, 
mientras que un R^2 bajo indica que la línea recta no explica bien la variabilidad en los datos.
En este caso, el R^2 es de 0.67, lo que indica que la línea recta explica una cantidad moderada de la variabilidad en los datos.
"""
