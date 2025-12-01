import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
#CONEXIÓN Y EXTRACCIÓN DE DATOS DE BAJA ROTACIÓN

server_name = '.'
database_name = 'Poly'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"

# Esta consulta identifica los 20 productos que se han vendido en el menor número
# de días distintos durante todo el periodo.
sql_query_baja_rotacion = """
SELECT TOP 20
    p.nombre AS Producto,
    SUM(v.cantidad) AS UnidadesTotalesVendidas,
    COUNT(DISTINCT v.TiempoKey) AS DiasDeVenta
FROM
    dbo.FactVentas v
JOIN
    dbo.DimProductos p ON v.ProductoKey = p.ProductoKey
GROUP BY
    p.nombre
ORDER BY
    DiasDeVenta ASC, -- Ordenamos por los que tienen menos días de venta
    UnidadesTotalesVendidas ASC;
"""

try:
    print("Conectando al Data Warehouse para análisis de baja rotación...")
    cnxn = pyodbc.connect(connection_string)
    df_baja_rotacion = pd.read_sql(sql_query_baja_rotacion, cnxn)
    cnxn.close()
    print("¡Datos de productos de baja rotación extraídos exitosamente!")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    df_baja_rotacion = pd.DataFrame()

# GENERACIÓN DEL REPORTE DE BAJA ROTACIÓN
if not df_baja_rotacion.empty:

    print("\n--- REPORTE DE PRODUCTOS DE BAJA ROTACIÓN (CANDIDATOS A LIQUIDACIÓN) ---")
    print(
        "Los siguientes 20 productos son los que se han vendido con menor frecuencia (en días distintos) durante los dos años de operación.\n")
    for index, row in df_baja_rotacion.iterrows():
        print(
            f"- {row['Producto']}: Se vendieron {row['UnidadesTotalesVendidas']} unidades en solo {row['DiasDeVenta']} días diferentes.")

    print("\n\nGenerando reporte visual de baja rotación...")
    plt.figure(figsize=(12, 10))

    ax = sns.barplot(
        data=df_baja_rotacion,
        x='UnidadesTotalesVendidas',
        y='Producto',
        palette='autumn_r',
        hue='Producto',
        legend=False
    )

    ax.bar_label(ax.containers[0], fmt='{:,.0f}', padding=5, fontsize=10, color='black')
    plt.title('Top 20 Productos con Menor Frecuencia de Venta (por Volumen)', fontsize=16)
    plt.xlabel('Total de Unidades Vendidas en 2 Años', fontsize=12)
    plt.ylabel('Producto', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.xlim(right=df_baja_rotacion['UnidadesTotalesVendidas'].max() * 1.15)

    plt.tight_layout()
    plt.show()

else:
    print("No se encontraron datos para el reporte de baja rotación.")

'''Resumen en Consola: Te mostrará una lista de los 20 productos menos dinámicos. Para cada uno, te dirá el total de unidades que vendió y, lo más importante, en cuántos días distintos se registró al menos una venta de ese producto.

Gráfico de Barras: Verás un gráfico de los "peores" 20 productos. Las barras más cortas representarán a los productos que son verdaderamente "fantasmas" en tus registros de ventas.
'''