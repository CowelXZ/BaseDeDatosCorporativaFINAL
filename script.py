import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CONFIGURACIÓN DE LA CONEXIÓN A LA BASE DE DATOS
#    Rellena estos datos con la información de tu servidor.
# =============================================================================

# El nombre del servidor que descubrimos antes.
server_name = '.'
# El nombre de la base de datos de tu proyecto.
database_name = 'Poly'

# Esta es la "dirección" que usa Python.
# Usamos "Trusted_Connection=yes" porque te conectas con tu usuario de Windows.
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"

# =============================================================================
# 2. LEER LOS DATOS DE LA TABLA DE RESULTADOS
#    (No ejecutes esta parte hasta que el script de SQL termine)
# =============================================================================

# La consulta para traer los patrones de compra.
sql_query = "SELECT * FROM dbo.ResultadosPatronesCompra"

print("Conectando a la base de datos...")
# Usaremos un bloque try/except para manejar errores de conexión.
try:
    # Establecemos la conexión
    cnxn = pyodbc.connect(connection_string)
    print("¡Conexión exitosa!")

    # Leemos los datos y los cargamos en un DataFrame de pandas
    df_patrones = pd.read_sql(sql_query, cnxn)

    # Cerramos la conexión
    cnxn.close()

    print("Datos cargados correctamente. Aquí tienes una muestra:")
    # Mostramos las 5 primeras filas de nuestros resultados
    print(df_patrones.head())

except pyodbc.Error as ex:
    sqlstate = ex.args[0]
    print(f"Error de conexión: {sqlstate}")
    print("Asegúrate de que el nombre del servidor y la base de datos son correctos.")

# =============================================================================
# 3. AQUÍ IRÁN NUESTROS GRÁFICOS (¡próximamente!)
# =============================================================================
# =============================================================================
# 3. VISUALIZACIÓN DE LOS PATRONES DE COMPRA MÁS FUERTES
# =============================================================================

# Asegurémonos de que el DataFrame no esté vacío
if not df_patrones.empty:

    # Ordenamos el DataFrame por la columna 'lift' de mayor a menor
    df_top_lift = df_patrones.sort_values(by='lift', ascending=False).head(15)

    # Creamos una nueva columna para las etiquetas del gráfico, combinando los dos productos
    df_top_lift['par_productos'] = df_top_lift['producto_A'] + '  ->  ' + df_top_lift['producto_B']

    # Configuramos el tamaño del gráfico para que se vea bien
    plt.figure(figsize=(12, 10))


    # Versión corregida y moderna
    sns.barplot(x='lift', y='par_productos', data=df_top_lift, hue='par_productos', palette='viridis', legend=False)



    # Añadimos títulos y etiquetas para que sea fácil de entender
    plt.title('Top 15 Pares de Productos con Mayor Lift', fontsize=16)
    plt.xlabel('Lift (Fuerza de la Asociación)', fontsize=12)
    plt.ylabel('Par de Productos', fontsize=12)

    # Ajustamos el layout y mostramos el gráfico
    plt.tight_layout()
    plt.show()

else:
    print("El DataFrame está vacío. No se pueden generar gráficos.")

    # =============================================================================
    # 4. VISUALIZACIÓN DE LOS PATRONES DE COMPRA MÁS CONFIABLES
    # =============================================================================

    # Asegurémonos de que el DataFrame no esté vacío
    if not df_patrones.empty:

        # Ordenamos el DataFrame por la columna 'confianza_A_hacia_B' de mayor a menor
        df_top_confidence = df_patrones.sort_values(by='confianza_A_hacia_B', ascending=False).head(15)

        # Creamos la etiqueta para el gráfico
        df_top_confidence['par_productos'] = df_top_confidence['producto_A'] + '  ->  ' + df_top_confidence[
            'producto_B']

        # Configuramos el tamaño del gráfico
        plt.figure(figsize=(12, 10))

        # Creamos el gráfico de barras usando la sintaxis corregida
        sns.barplot(x='confianza_A_hacia_B', y='par_productos', data=df_top_confidence, hue='par_productos',
                    palette='plasma', legend=False)

        # Añadimos títulos y etiquetas
        plt.title('Top 15 Pares de Productos con Mayor Confianza', fontsize=16)
        plt.xlabel('Confianza (Probabilidad de compra conjunta)', fontsize=12)
        plt.ylabel('Regla de Asociación (Si compra A -> compra B)', fontsize=12)

        # Ajustamos el layout y mostramos el gráfico
        plt.tight_layout()
        plt.show()

    else:
        print("El DataFrame está vacío. No se pueden generar gráficos.")
