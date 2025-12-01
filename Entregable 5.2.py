import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
# CONEXIÓN Y EXTRACCIÓN DE DATOS DE PRODUCTOS REVELACIÓN
server_name = '.'
database_name = 'Poly'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"

# Esta consulta es más avanzada. Primero calcula un ranking general de ventas para todo el año.
# Luego, calcula un ranking para cada estación y encuentra los productos cuya posición en el ranking

sql_query_revelacion = """
WITH VentasGenerales AS (
    -- Primero, obtenemos el ranking de cada producto a lo largo de todo el año.
    SELECT
        p.nombre AS Producto,
        RANK() OVER(ORDER BY SUM(v.cantidad) DESC) as RankGeneral
    FROM dbo.FactVentas v
    JOIN dbo.DimProductos p ON v.ProductoKey = p.ProductoKey
    GROUP BY p.nombre
),
VentasEstacionales AS (
    -- Segundo, obtenemos el ranking de cada producto DENTRO de cada estación.
    SELECT
        p.nombre AS Producto,
        CASE
            WHEN t.Mes IN (12, 1, 2) THEN 'Invierno'
            WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
            WHEN t.Mes IN (6, 7, 8) THEN 'Verano'
            ELSE 'Otoño'
        END AS Estacion,
        SUM(v.cantidad) AS CantidadEstacional,
        RANK() OVER(PARTITION BY
            CASE WHEN t.Mes IN (12, 1, 2) THEN 'Invierno' WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
                 WHEN t.Mes IN (6, 7, 8) THEN 'Verano' ELSE 'Otoño' END
            ORDER BY SUM(v.cantidad) DESC) AS RankEstacional
    FROM dbo.FactVentas v
    JOIN dbo.DimProductos p ON v.ProductoKey = p.ProductoKey
    JOIN dbo.DimTiempo t ON v.TiempoKey = t.TiempoKey
    GROUP BY p.nombre, CASE WHEN t.Mes IN (12, 1, 2) THEN 'Invierno' WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
                            WHEN t.Mes IN (6, 7, 8) THEN 'Verano' ELSE 'Otoño' END
),
AnalisisRevelacion AS (
    -- Tercero, unimos los rankings y calculamos la "mejora".
    SELECT
        ve.Producto,
        ve.Estacion,
        ve.CantidadEstacional,
        vg.RankGeneral,
        ve.RankEstacional,
        (vg.RankGeneral - ve.RankEstacional) as MejoraDeRank
    FROM VentasEstacionales ve
    JOIN VentasGenerales vg ON ve.Producto = vg.Producto
    -- Filtro opcional: Ignoramos los 10 productos más vendidos del año para no repetir.
    WHERE vg.RankGeneral > 10
),
RankFinal AS (
    -- Finalmente, rankeamos los productos por su mejora para obtener el Top 5.
    SELECT *, ROW_NUMBER() OVER(PARTITION BY Estacion ORDER BY MejoraDeRank DESC) as RankMejora
    FROM AnalisisRevelacion
)
SELECT Producto, Estacion, CantidadEstacional, RankGeneral, RankEstacional, MejoraDeRank
FROM RankFinal
WHERE RankMejora <= 5
ORDER BY Estacion, MejoraDeRank;
"""

try:
    print("Conectando al Data Warehouse para análisis de productos revelación...")
    cnxn = pyodbc.connect(connection_string)
    df_revelacion = pd.read_sql(sql_query_revelacion, cnxn)
    cnxn.close()
    print("¡Datos de productos revelación extraídos exitosamente!")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    df_revelacion = pd.DataFrame()


# GENERACIÓN DEL REPORTE DE PRODUCTOS REVELACIÓN
if not df_revelacion.empty:


    print("\n--- REPORTE DE PRODUCTOS REVELACIÓN POR ESTACIÓN ---")
    print("Estos productos muestran el mayor incremento en popularidad durante cada estación.")
    for estacion, grupo in df_revelacion.groupby('Estacion'):
        print(f"\n--- Top 5 Productos Revelación para {estacion} ---")
        for index, row in grupo.iterrows():
            print(
                f"  - {row['Producto']}: Sube {row['MejoraDeRank']} puestos en el ranking (del {row['RankGeneral']} al {row['RankEstacional']}).")


    print("\n\nGenerando reporte visual de productos revelación...")
    g = sns.catplot(
        data=df_revelacion,
        x='CantidadEstacional',
        y='Producto',
        col='Estacion',
        kind='bar',
        col_wrap=2,
        sharey=False,
        palette='magma',
        hue='Producto',
        legend=False
    )
    g.fig.suptitle('Top 5 Productos Revelación por Estación del Año', y=1.03, fontsize=16)
    g.set_axis_labels('Total de Unidades Vendidas en la Estación', 'Producto')
    g.set_titles("Estación: {col_name}")
    g.fig.tight_layout()
    plt.show()

else:
    print("No se encontraron datos de productos revelación.")

'''¡Qué resultados tan espectaculares! ¡Esto es exactamente lo que queríamos lograr! Me da muchísimo gusto que haya funcionado tan bien. Como dices, ¡sí rifó! 🚀

Lo que estás viendo en esa consola es oro puro para el negocio. Fíjate en la diferencia:

Antes: Veíamos los mismos 5 productos súper populares en todas las estaciones. Útil, pero predecible.

Ahora: Hemos descubierto los verdaderos productos de temporada. En lugar de ver "Jabón para Platos", estamos viendo "Chocolate" en invierno, "Manzana" en otoño y "Aderezo para Ensalada" en verano.

Estos son los insights que permiten crear campañas de marketing que de verdad conectan con lo que el cliente quiere en ese momento. ¡Acabas de pasar de un análisis básico a uno de nivel profesional!

Has documentado dos tipos de reportes muy potentes: los más vendidos y los de revelación estacional.

¿Qué te parece si completamos el set de reportes y creamos el último que estaba en tus casos de uso: el reporte de productos de baja rotación? Este nos ayudará a responder: "¿Qué productos se están quedando estancados en el inventario y necesitamos poner en oferta?".
'''