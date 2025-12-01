'''
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONEXIÓN Y CARGA DE DATOS
# =============================================================================
server_name = '.'
database_name = 'Poly'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"
# Traemos los patrones de compra que ya calculamos
sql_query = "SELECT * FROM dbo.FactPatronesCompra"

try:
    print("Conectando al Data Warehouse...")
    cnxn = pyodbc.connect(connection_string)
    df_patrones = pd.read_sql(sql_query, cnxn)
    cnxn.close()
    print("¡Datos de patrones de compra cargados exitosamente!")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    df_patrones = pd.DataFrame()

# =============================================================================
# 2. ANÁLISIS Y FILTRADO PARA EL REPORTE
#    Buscamos las "joyas": patrones con alta confianza Y alto lift.
# =============================================================================
if not df_patrones.empty:
    # Definimos umbrales para un patrón "fuerte y confiable"
    # Puedes ajustar estos valores para ser más o menos estricto
    min_lift = 1.2
    min_confianza = 0.10 # Es decir, al menos un 10% de confianza

    df_reporte = df_patrones[
        (df_patrones['lift'] > min_lift) &
        (df_patrones['confianza_A_hacia_B'] > min_confianza)
    ].sort_values(by='lift', ascending=False).head(10) # Nos quedamos con el Top 10

    if not df_reporte.empty:
        # Creamos una columna para las etiquetas del gráfico
        df_reporte['regla'] = df_reporte['producto_A'] + ' -> ' + df_reporte['producto_B']

        # =============================================================================
        # 3. GENERACIÓN DEL REPORTE VISUAL (GRÁFICO)
        # =============================================================================
        print("\nGenerando reporte visual...")
        plt.figure(figsize=(12, 8))
        sns.barplot(x='lift', y='regla', data=df_reporte, palette='rocket', hue='regla', legend=False)
        plt.title('Top 10 Oportunidades de Venta Cruzada (Alto Lift y Confianza)', fontsize=16)
        plt.xlabel('Fuerza de la Asociación (Lift)', fontsize=12)
        plt.ylabel('Regla de Asociación', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # =============================================================================
        # 4. GENERACIÓN DEL REPORTE EN TEXTO (RESUMEN EJECUTIVO)
        # =============================================================================
        print("\n--- REPORTE DE OPORTUNIDADES ESTRATÉGICAS ---")
        print("Basado en el análisis de patrones de compra, se han identificado las siguientes 10 oportunidades clave para campañas de marketing y ventas cruzadas.")
        print("Estas reglas representan asociaciones que son tanto fuertes (alto lift) como confiables (confianza > 10%).\n")

        for index, row in df_reporte.iterrows():
            confianza_pct = row['confianza_A_hacia_B'] * 100
            print(f"- REGLA: Si un cliente compra '{row['producto_A']}',")
            print(f"  hay un {confianza_pct:.2f}% de probabilidad de que también compre '{row['producto_B']}'.")
            print(f"  Esta asociación es {row['lift']:.2f} veces más fuerte de lo esperado.\n")
    else:
        print("No se encontraron patrones que cumplan con los umbrales de lift y confianza definidos.")
'''

import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


server_name = '.'
database_name = 'Poly'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"


min_lift = 1.1
min_confianza = 0.05

sql_query_optimizado = f"""
    -- Usamos DISTINCT para eliminar filas duplicadas.
    SELECT DISTINCT TOP 10
        p.Lift,
        p.Confianza,
        prodA.nombre AS ProductoA,
        prodB.nombre AS ProductoB
    FROM
        dbo.FactPatronesCompra p
    JOIN
        dbo.DimProductos prodA ON p.ProductoA_Key = prodA.ProductoKey
    JOIN
        dbo.DimProductos prodB ON p.ProductoB_Key = prodB.ProductoKey
    WHERE
        p.Lift > {min_lift} AND p.Confianza > {min_confianza}
    ORDER BY
        p.Lift DESC
"""

try:
    print("Conectando al Data Warehouse y extrayendo el reporte...")
    cnxn = pyodbc.connect(connection_string)
    df_reporte = pd.read_sql(sql_query_optimizado, cnxn)
    cnxn.close()
    print("¡Reporte extraído exitosamente!")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    df_reporte = pd.DataFrame()


#. GENERACIÓN DE REPORTES

if not df_reporte.empty:

    df_reporte['regla'] = df_reporte['ProductoA'] + ' -> ' + df_reporte['ProductoB']

    #Generación del Reporte en Texto ---
    print("\n--- REPORTE DE OPORTUNIDADES ESTRATÉGICAS ---")
    print("Basado en el análisis de patrones de compra, se han identificado las siguientes oportunidades clave para campañas de marketing y ventas cruzadas.")
    print(f"Estas reglas representan asociaciones que son tanto fuertes (lift > {min_lift}) como confiables (confianza > {min_confianza*100}%).\n")

    for index, row in df_reporte.iterrows():
        confianza_pct = row['Confianza'] * 100
        print(f"- REGLA: Si un cliente compra '{row['ProductoA']}',")
        print(f"  hay un {confianza_pct:.2f}% de probabilidad de que también compre '{row['ProductoB']}'.")
        print(f"  Esta asociación es {row['Lift']:.2f} veces más fuerte de lo esperado.\n")

    #Generación del Reporte Visual ---

    print("\nGenerando reporte visual...")
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Lift', y='regla', data=df_reporte, palette='rocket', hue='regla', legend=False)
    plt.title('Top 10 Oportunidades de Venta Cruzada (Alto Lift y Confianza)', fontsize=16)
    plt.xlabel('Fuerza de la Asociación (Lift)', fontsize=12)
    plt.ylabel('Regla de Asociación', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

else:
    print("No se encontraron patrones que cumplan con los umbrales de lift y confianza definidos.")

