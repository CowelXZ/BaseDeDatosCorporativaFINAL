import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# =============================================================================
# 1. CONFIGURACIÓN Y CONEXIÓN A LA BASE DE DATOS
# =============================================================================

# -- Rellena con tus datos --
server_name = '.'
database_name = 'Poly'

# -- Cadena de conexión --
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"
sql_query = "SELECT * FROM dbo.ResultadosPatronesCompra"

# -- Cargar los datos en un DataFrame de Pandas --
try:
    print("Conectando a la base de datos...")
    cnxn = pyodbc.connect(connection_string)
    df_patrones = pd.read_sql(sql_query, cnxn)
    cnxn.close()
    print("¡Conexión y carga de datos exitosas!")
    print(f"Se han cargado {len(df_patrones)} patrones de compra.")
except Exception as e:
    print(f"Error al conectar o cargar los datos: {e}")
    df_patrones = pd.DataFrame()  # Creamos un DataFrame vacío en caso de error

# =============================================================================
# 2. ANÁLISIS INTERACTIVO: ¿QUÉ SE VENDE CON...?
#    ¡Aquí puedes hacerle preguntas a tus datos!
# =============================================================================
if not df_patrones.empty:
    # -- Modifica este nombre de producto para explorar --
    producto_a_investigar = 'Banana'

    print(f"\n--- Analizando qué productos se compran con '{producto_a_investigar}' ---")

    # Filtramos el DataFrame para encontrar las reglas donde aparece nuestro producto
    reglas_producto = df_patrones[df_patrones['producto_A'] == producto_a_investigar].sort_values(
        by='confianza_A_hacia_B', ascending=False)

    if not reglas_producto.empty:
        print(f"Top 5 productos comprados junto con '{producto_a_investigar}' (ordenados por confianza):")
        # Mostramos los 5 resultados más relevantes
        print(reglas_producto[['producto_B', 'confianza_A_hacia_B', 'lift']].head())
    else:
        print(f"No se encontraron reglas de compra significativas para '{producto_a_investigar}'.")

# =============================================================================
# 3. VISUALIZACIÓN AVANZADA: GRAFO DE RED DE PRODUCTOS
#    Muestra las conexiones entre los productos más fuertes.
# =============================================================================
if not df_patrones.empty:
    print("\n--- Generando Grafo de Red de Productos ---")

    # Filtramos para quedarnos solo con las relaciones más fuertes y hacer el gráfico legible
    # Puedes ajustar este valor de 'lift' para ver más o menos conexiones
    df_graph = df_patrones[df_patrones['lift'] > .5].head(50)
    #
    ##Aca se cambia el valor de lift de arribita para ver mejores ppatrones
    #
    #
    if not df_graph.empty:
        # Creamos el grafo a partir del DataFrame
        G = nx.from_pandas_edgelist(df_graph,
                                    source='producto_A',
                                    target='producto_B',
                                    edge_attr='lift')  # Usamos el lift como "peso" de la conexión

        # Preparamos la visualización
        plt.figure(figsize=(18, 18))
        pos = nx.spring_layout(G, k=0.7)  # 'k' ajusta la separación entre nodos

        # Dibujamos los nodos (productos)
        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='skyblue')

        # Dibujamos las etiquetas de los nodos
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Dibujamos las aristas (conexiones)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')

        # Añadimos etiquetas a las aristas para mostrar el valor de lift
        edge_labels = nx.get_edge_attributes(G, 'lift')
        formatted_edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_size=8)

        # Configuramos y mostramos el gráfico
        plt.title('Grafo de Red de Asociaciones de Productos (Lift > 1)', size=20)
        plt.axis('off')  # Ocultamos los ejes
        plt.show()
    else:
        print(
            "No se encontraron suficientes relaciones fuertes para generar el grafo. Intenta bajar el umbral de 'lift'.")
