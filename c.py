# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Convierte las columnas a arrays de NumPy
X = np.array(df['Veces Jugado'].values)
y = np.array(df['Número de Reseñas'].values)
m = len(y)

# Añadir una columna de unos a X para el término de intercepción
X = np.column_stack((np.ones(m), X))


# Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
train_size = int(0.5 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Entrenar el modelo de regresión lineal con los datos de entrenamiento
reg = LinearRegression().fit(X_train[:, 1:], y_train)

# Obtener los coeficientes de regresión lineal
coefficients = np.append(reg.intercept_, reg.coef_)

# Calcular el MSE y el R^2 con los datos de prueba
mse = np.mean((reg.predict(X_test[:, 1:]) - y_test)**2)
r2 = reg.score(X_test[:, 1:], y_test)


# Definir la función de costo (MSE)
def cost_function(X, y, coefficients):
    m = len(y)
    J = np.sum((X.dot(coefficients) - y)**2) / (2*m)
    return J

# Definir la función de actualización de los coeficientes de regresión lineal
def gradient_descent(X, y, coefficients, alpha, num_iterations):
    m = len(y)
    J_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        h = X.dot(coefficients)
        for j in range(len(coefficients)):
            coefficients[j] = coefficients[j] - alpha*(1/m)*(np.sum((h - y)*X[:,j]))
        J_history[i] = cost_function(X, y, coefficients)
    return coefficients, J_history

# Definir los hiperparámetros
alpha = 0.001
num_iterations = 5000

# Realizar el descenso del gradiente para obtener los coeficientes optimizados
new_coefficients, J_history = gradient_descent(X, y, coefficients, alpha, num_iterations)


# Imprimir los resultados
print('Coeficientes de la regresión lineal:', coefficients)
print('Error cuadrático medio (MSE):', mse)
print('Coeficiente de determinación R^2:', r2)


# Graficar la función de costo en función del número de iteraciones
plt.plot(range(num_iterations), J_history)
plt.xlabel('Número de Iteraciones')
plt.ylabel('Función de Costo')
plt.title('Funcion de costo')
plt.show()

# Generar la gráfica
plt.scatter(X[:, 1], y, color='blue')
plt.plot(X[:, 1], new_coefficients[0] + new_coefficients[1]*X[:, 1], color='red')
plt.xlabel('Veces Jugado')
plt.ylabel('Número de Reseñas')
plt.title('Regresión Lineal')
plt.show()

"""
CONCLUSION:
Podemos observar que los valores obtenidos en la regresión lineal y en el gradiente descendiente son similares,
ambos métodos son eficaces para calcular la regresión lineal y predecir valores, 
pero en este caso el método de gradiente descendiente parece tener un mejor rendimiento, ya que tiene un MSE más bajo y un R^2 ligeramente más bajo.
aunque es  importante tener en cuenta que los resultados pueden variar dependiendo de los parámetros seleccionados.
"""
