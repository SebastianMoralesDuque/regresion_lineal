import pandas as pd

# Lee el archivo CSV
df = pd.read_csv('games.csv', delimiter=',')

# Cambia los nombres de las columnas al español
df.columns = ['#', 'Título', 'Fecha de Lanzamiento', 'Equipo', 'Calificación', 'Veces Listado', 'Número de Reseñas', 'Géneros', 'Resumen', 'Reseñas', 'Veces Jugado', 'Jugando', 'Pendientes', 'Lista de Deseos']

# Exploración inicial del dataframe
print(df.info())  # Imprime información sobre el dataframe, como el número de filas, columnas y tipos de datos de cada columna
print(df.head())  # Imprime las primeras 5 filas del dataframe


