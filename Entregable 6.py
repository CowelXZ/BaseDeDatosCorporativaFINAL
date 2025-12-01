import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import numpy as np

# Ignoramos advertencias para una salida más limpia
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONEXIÓN Y CARGA DE DATOS
# =============================================================================

server_name = '.'
database_name = 'Poly'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"
sql_query = "SELECT TOP 100000 * FROM dbo.FactPerfilesDeCompra"

try:
    print("Conectando a la base de datos y cargando el set de entrenamiento...")
    cnxn = pyodbc.connect(connection_string)
    df = pd.read_sql(sql_query, cnxn)
    cnxn.close()
    print(f"¡Carga exitosa! Se analizarán {len(df)} tickets.")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    df = pd.DataFrame()

if not df.empty:
    df_tickets = df['id_ticket']
    df_gastos = df.drop('id_ticket', axis=1)

    # =============================================================================
    # 2. PREPARACIÓN DE DATOS (ESCALADO)
    # =============================================================================
    print("\nEstandarizando los datos...")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_gastos)

    # No es necesario volver a mostrar el gráfico del codo, ya lo conocemos.

    # =============================================================================
    # 3. ENTRENAR EL MODELO K-MEANS CON EL NUEVO k
    #    Aquí está la magia. Simplemente cambiamos el número de clusters a buscar.
    # =============================================================================

    # ---- ¡AQUÍ ESTÁ EL CAMBIO! ----
    # Probemos con 5 para ver si descubrimos perfiles más detallados.
    # ¡Puedes cambiar este número a 6, 7, etc., para seguir explorando!
    k_nuevo = 5

    print(f"\nRe-entrenando el modelo K-Means con k={k_nuevo}...")

    kmeans = KMeans(n_clusters=k_nuevo, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    df['cluster'] = kmeans.labels_

    # =============================================================================
    # 4. INTERPRETAR LOS NUEVOS PERFILES DE COMPRA
    # =============================================================================
    print(f"\n--- {k_nuevo} Perfiles de Compra Descubiertos ---")
    perfiles = df.groupby('cluster').mean().drop('id_ticket', axis=1)
    perfiles['tamaño_cluster'] = df['cluster'].value_counts()
    print(perfiles)

    # =============================================================================
    # 5. VISUALIZAR LOS NUEVOS PERFILES CON GRÁFICOS DE RADAR
    # =============================================================================
    print("\nGenerando gráficos de radar para cada perfil...")

    labels = perfiles.columns[:-1]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))

    for i, row in perfiles.iterrows():
        data = row[:-1].tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=2, linestyle='solid', label=f"Perfil {i} ({row['tamaño_cluster']} tickets)")
        ax.fill(angles, data, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(f"{k_nuevo} Perfiles de Compra por Gasto Promedio", size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout(pad=2.0)
    plt.show()
