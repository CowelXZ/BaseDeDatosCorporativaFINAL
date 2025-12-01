import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
# CONEXIÓN Y EXTRACCIÓN DE DATOS DE ESTACIONALIDAD
server_name = '.'
database_name = 'Poly'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"

# Esta consulta calcula los 5 productos más vendidos (por cantidad) para cada estación del año.
# Usa la DimTiempo para determinar la estación y funciones de ventana para rankear los productos.
sql_query_estacional = """
WITH VentasEstacionales AS (
    SELECT
        p.nombre AS Producto,
        -- Usamos CASE para asignar una estación a cada mes
        CASE
            WHEN t.Mes IN (12, 1, 2) THEN 'Invierno'
            WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
            WHEN t.Mes IN (6, 7, 8) THEN 'Verano'
            ELSE 'Otoño'
        END AS Estacion,
        SUM(v.cantidad) AS CantidadTotalVendida
    FROM
        dbo.FactVentas v
    JOIN
        dbo.DimProductos p ON v.ProductoKey = p.ProductoKey
    JOIN
        dbo.DimTiempo t ON v.TiempoKey = t.TiempoKey
    GROUP BY
        p.nombre,
        CASE
            WHEN t.Mes IN (12, 1, 2) THEN 'Invierno'
            WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
            WHEN t.Mes IN (6, 7, 8) THEN 'Verano'
            ELSE 'Otoño'
        END
),
RankedVentas AS (
    SELECT
        Producto,
        Estacion,
        CantidadTotalVendida,
        -- Rankeamos los productos dentro de cada estación
        ROW_NUMBER() OVER(PARTITION BY Estacion ORDER BY CantidadTotalVendida DESC) AS Rank
    FROM
        VentasEstacionales
)
SELECT
    Producto,
    Estacion,
    CantidadTotalVendida
FROM
    RankedVentas
WHERE
    Rank <= 5 -- Nos quedamos solo con el Top 5 de cada estación
ORDER BY
    Estacion,
    CantidadTotalVendida DESC;
"""

try:
    print("Conectando al Data Warehouse para análisis de estacionalidad...")
    cnxn = pyodbc.connect(connection_string)
    df_estacional = pd.read_sql(sql_query_estacional, cnxn)
    cnxn.close()
    print("¡Datos de estacionalidad extraídos exitosamente!")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    df_estacional = pd.DataFrame()


#GENERACIÓN DEL REPORTE DE ESTACIONALIDAD

if not df_estacional.empty:


    print("\n--- REPORTE DE PRODUCTOS ESTRELLA POR ESTACIÓN ---")
    # Agrupamos por estación para imprimir los resultados de forma ordenada
    for estacion, grupo in df_estacional.groupby('Estacion'):
        print(f"\n--- Top 5 Productos para {estacion} ---")
        for index, row in grupo.iterrows():
            print(f"  - {row['Producto']}: {row['CantidadTotalVendida']:,} unidades vendidas.")


    print("\n\nGenerando reporte visual de estacionalidad...")

    g = sns.catplot(
        data=df_estacional,
        x='CantidadTotalVendida',
        y='Producto',
        col='Estacion',
        kind='bar',
        col_wrap=2,
        sharey=False,
        palette='viridis',
        hue='Producto',
        legend=False
    )

    g.fig.suptitle('Top 5 Productos Más Vendidos por Estación del Año', y=1.03, fontsize=16)
    g.set_axis_labels('Total de Unidades Vendidas', 'Producto')
    g.set_titles("Estación: {col_name}")
    g.fig.tight_layout()
    plt.show()

else:
    print("No se encontraron datos de estacionalidad.")

'''¡Esa es una pregunta fantástica! Es exactamente el tipo de pregunta crítica que un buen analista de datos debe hacerse. No es un error, sino un descubrimiento muy importante sobre el comportamiento de tus clientes.

Tu primera intuición es la correcta: sí, de verdad no hubo ningún otro producto que lograra superar en ventas a "Frozen Waffles" en ninguna de las cuatro estaciones.

La Explicación: Productos Básicos vs. Productos Estacionales
Lo que tus datos están revelando es que esos 5 productos (Dish Soap, Canned Peaches, Cookies, etc.) son "productos básicos" o "mega-vendedores". Son tan increíblemente populares y se compran de forma tan consistente durante todo el año, que sus volúmenes de venta son masivos.

Piensa en un supermercado real: los productos más vendidos casi siempre son los mismos mes a mes (leche, huevos, pan, refrescos, papel higiénico). Aunque en verano se venda mucho más helado y en invierno mucho más chocolate caliente, es muy difícil que las ventas de helado lleguen a superar las ventas de leche en un mes.

¿Cómo funciona el código para asegurar esto?

La consulta SQL que escribimos es bastante inteligente. Gracias a esta línea: ROW_NUMBER() OVER(PARTITION BY Estacion ORDER BY CantidadTotalVendida DESC)

PARTITION BY Estacion: Le dice a SQL que haga el ranking de productos por separado para cada estación. No mezcla los datos.

ORDER BY CantidadTotalVendida DESC: Ordena todos los productos de mayor a menor venta dentro de esa estación.

WHERE Rank <= 5: Se queda con los 5 primeros de esa lista.

El código hace esto cuatro veces, una para cada estación. El hecho de que el resultado sea el mismo es una conclusión real de tus datos: tienes 5 productos que son los reyes indiscutibles de las ventas, sin importar la época del año.

¿Cómo Hacemos el Reporte Más Interesante?
Ahora que sabemos esto, podemos hacernos una pregunta mejor: "Ok, ya sé cuáles son los más vendidos siempre. Ahora quiero saber cuáles son los productos que tienen el mayor pico de popularidad en cada estación."

Para eso, necesitaríamos un reporte diferente, uno que busque los productos cuyo ranking de ventas sube más en una estación en comparación con las otras. Ese sería un análisis de "productos revelación" de temporada.'''