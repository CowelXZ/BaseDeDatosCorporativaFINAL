import streamlit as st
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y CONEXIÓN A LA BD
# =============================================================================
st.set_page_config(page_title="Dashboard 'Poly'", layout="wide")

# --- Configuración de la Conexión a la BD ---
DB_CONFIG = {
    "server_name": '.',
    "database_name": 'Poly',
    "connection_string": "DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=Poly;Trusted_Connection=yes;"
}


# =============================================================================
# OPTIMIZACIÓN DEL BACKEND: CACHING DE DATOS
# =============================================================================
@st.cache_data
def cargar_datos_sql(query):
    """Función segura para conectarse y cargar datos en un DataFrame."""
    try:
        cnxn = pyodbc.connect(DB_CONFIG["connection_string"])
        df = pd.read_sql(query, cnxn)
        cnxn.close()
        return df
    except Exception as e:
        st.error(f"Error al conectar o consultar la base de datos: {e}")
        return pd.DataFrame()


# =============================================================================
# PUNTO 8: GOBERNANZA Y AUTENTICACIÓN
# =============================================================================
USUARIOS_REGISTRADOS = {
    "gerente": {"password": "123", "role": "admin"},
    "cajero": {"password": "456", "role": "user"}
}


def check_login(username, password):
    """Verifica credenciales y actualiza el estado de sesión."""
    if username in USUARIOS_REGISTRADOS and USUARIOS_REGISTRADOS[username]["password"] == password:
        st.session_state['logged_in'] = True
        st.session_state['role'] = USUARIOS_REGISTRADOS[username]["role"]
        st.session_state['username'] = username
        st.sidebar.success(f"¡Bienvenido, {st.session_state['username']}!")
    else:
        st.sidebar.error("Usuario o contraseña incorrectos.")


def show_login_page():
    """Muestra el formulario de inicio de sesión en la barra lateral."""
    st.sidebar.title("Iniciar Sesión")
    username = st.sidebar.text_input("Usuario", key="login_user")
    password = st.sidebar.text_input("Contraseña", type="password", key="login_pass")

    if st.sidebar.button("Entrar", key="login_button"):
        check_login(username, password)


# =============================================================================
# DEFINICIÓN DE PÁGINAS DEL DASHBOARD (COMPLETAS Y CORREGIDAS)
# =============================================================================

def show_page_home():
    """Página de bienvenida."""
    st.title("Bienvenido al Dashboard de Inteligencia de Negocios 'Poly'")
    st.markdown("---")
    if st.session_state['role'] == 'admin':
        st.subheader("Usted ha iniciado sesión como Administrador.")
        st.write("Utilice el menú de navegación de la izquierda para explorar los diferentes análisis y reportes:")
        st.markdown("""
        * **Análisis de Patrones de Compra:** Descubre qué productos se compran juntos (Market Basket Analysis).
        * **Análisis de Estacionalidad:** Ve los productos estrella y revelación de cada temporada.
        * **Análisis de Baja Rotación:** Identifica productos que no se están vendiendo.
        * **IA: Perfiles de Cliente (K-Means):** Descubre los 5 tipos de compradores en la tienda.
        * **IA: Clasificador de Clientes (SVM):** Evalúa el rendimiento de nuestro modelo predictivo.
        """)
    else:
        st.subheader(f"Bienvenido, {st.session_state['username']}.")
        st.write("Actualmente tiene permisos de usuario estándar y solo puede ver esta página de bienvenida.")
        st.warning("Contacte a un administrador para permisos de reportes.")


def show_page_patrones_compra():
    """Página para el reporte de Patrones de Compra (Entregable 5)."""
    st.title("Reporte: Oportunidades de Venta Cruzada (Patrones de Compra)")
    st.markdown(
        "Este reporte se conecta al Data Warehouse (`dbo.FactPatronesCompra`) y extrae las 10 principales oportunidades de marketing basadas en un alto **Lift** y **Confianza**.")

    # --- Configuración de Filtros ---
    st.sidebar.subheader("Filtros del Reporte")
    min_lift = st.sidebar.slider("Lift Mínimo:", 1.0, 2.0, 1.1, 0.05)
    min_confianza = st.sidebar.slider("Confianza Mínima:", 0.01, 0.3, 0.05, 0.01)

    # --- Lógica de Carga de Datos ---
    sql_query_optimizado = f"""
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

    with st.spinner("Consultando el Data Warehouse..."):
        df_reporte = cargar_datos_sql(sql_query_optimizado)  # Usamos la función de caché

    # --- Generación de Reportes ---
    if not df_reporte.empty:
        df_reporte['regla'] = df_reporte['ProductoA'] + ' -> ' + df_reporte['ProductoB']

        col1, col2 = st.columns([1, 1.5])

        # Columna 1: Reporte de Texto
        with col1:
            st.subheader("Oportunidades Estratégicas")
            st.write(f"Reglas con Lift > {min_lift} y Confianza > {min_confianza * 100}%.")
            for index, row in df_reporte.iterrows():
                confianza_pct = row['Confianza'] * 100
                st.markdown(f"**Regla:** Si compra `{row['ProductoA']}`:")
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;...hay un **{confianza_pct:.2f}%** de probabilidad de que compre `{row['ProductoB']}`.")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*(Relación {row['Lift']:.2f} veces más fuerte de lo esperado)*")
                st.divider()

        # Columna 2: Reporte Visual
        with col2:
            st.subheader("Visualización de la Fuerza (Lift)")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(ax=ax, x='Lift', y='regla', data=df_reporte, palette='rocket', hue='regla', legend=False)
            ax.set_title(f'Top {len(df_reporte)} Oportunidades de Venta Cruzada')
            ax.set_xlabel('Fuerza de la Asociación (Lift)')
            ax.set_ylabel('Regla de Asociación')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)  # Comando de Streamlit para dibujar el gráfico

    else:
        st.error("No se encontraron patrones que cumplan con los umbrales de lift y confianza definidos.")


def show_page_estacionalidad():
    """Página para el reporte de Estacionalidad (Entregable 5.1 y 5.2)."""
    st.title("Reporte: Análisis de Estacionalidad")

    reporte_tipo = st.selectbox("Seleccionar tipo de reporte:", ["Productos Estrella", "Productos Revelación"])

    if reporte_tipo == "Productos Estrella":
        st.subheader("Top 5 Productos Estrella por Estación")
        st.write("Muestra los 5 productos con el mayor volumen de ventas en cada estación.")

        # --- Lógica de Carga de Datos (Estrella) ---
        sql_query_estacional = """
            WITH VentasEstacionales AS (
                SELECT
                    p.nombre AS Producto,
                    CASE
                        WHEN t.Mes IN (12, 1, 2) THEN 'Invierno'
                        WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
                        WHEN t.Mes IN (6, 7, 8) THEN 'Verano'
                        ELSE 'Otoño'
                    END AS Estacion,
                    SUM(v.cantidad) AS CantidadTotalVendida
                FROM dbo.FactVentas v
                JOIN dbo.DimProductos p ON v.ProductoKey = p.ProductoKey
                JOIN dbo.DimTiempo t ON v.TiempoKey = t.TiempoKey
                GROUP BY p.nombre, CASE WHEN t.Mes IN (12, 1, 2) THEN 'Invierno' WHEN t.Mes IN (3, 4, 5) THEN 'Primavera' WHEN t.Mes IN (6, 7, 8) THEN 'Verano' ELSE 'Otoño' END
            ),
            RankedVentas AS (
                SELECT Producto, Estacion, CantidadTotalVendida,
                       ROW_NUMBER() OVER(PARTITION BY Estacion ORDER BY CantidadTotalVendida DESC) AS Rank
                FROM VentasEstacionales
            )
            SELECT Producto, Estacion, CantidadTotalVendida
            FROM RankedVentas
            WHERE Rank <= 5
            ORDER BY Estacion, CantidadTotalVendida DESC;
        """
        with st.spinner("Calculando reporte de productos estrella..."):
            df_estacional = cargar_datos_sql(sql_query_estacional)

        # --- Generación de Reporte (Estrella) ---
        if not df_estacional.empty:
            for estacion, grupo in df_estacional.groupby('Estacion'):
                with st.expander(f"Top 5 Productos para {estacion}"):
                    st.dataframe(grupo)

            st.subheader("Visualización Comparativa")
            # Creamos la figura explícitamente
            g = sns.catplot(
                data=df_estacional, x='CantidadTotalVendida', y='Producto',
                col='Estacion', kind='bar', col_wrap=2, sharey=False,
                palette='viridis', hue='Producto', legend=False
            )
            g.fig.suptitle('Top 5 Productos Más Vendidos por Estación del Año', y=1.03, fontsize=16)
            g.set_axis_labels('Total de Unidades Vendidas', 'Producto')
            g.set_titles("Estación: {col_name}")
            g.fig.tight_layout()
            st.pyplot(g.fig)  # Mostramos la figura en Streamlit
        else:
            st.error("No se pudieron cargar los datos de Productos Estrella.")

    else:  # Productos Revelación
        st.subheader("Top 5 Productos Revelación por Estación")
        st.write("Muestra los productos que más suben en el ranking de popularidad en cada estación.")

        # --- Lógica de Carga de Datos (Revelación) ---
        # CORRECCIÓN 1: Se completa la consulta SQL que estaba incompleta.
        sql_query_revelacion = """
            WITH VentasGenerales AS (
                SELECT p.nombre AS Producto,
                       RANK() OVER(ORDER BY SUM(v.cantidad) DESC) as RankGeneral
                FROM dbo.FactVentas v
                JOIN dbo.DimProductos p ON v.ProductoKey = p.ProductoKey
                GROUP BY p.nombre
            ),
            VentasEstacionales AS (
                SELECT p.nombre AS Producto,
                    CASE
                        WHEN t.Mes IN (12, 1, 2) THEN 'Invierno'
                        WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
                        WHEN t.Mes IN (6, 7, 8) THEN 'Verano'
                        ELSE 'Otoño'
                    END AS Estacion,
                    SUM(v.cantidad) AS CantidadEstacional,
                    RANK() OVER(PARTITION BY
                        CASE
                            WHEN t.Mes IN (12, 1, 2) THEN 'Invierno'
                            WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
                            WHEN t.Mes IN (6, 7, 8) THEN 'Verano'
                            ELSE 'Otoño'
                        END
                        ORDER BY SUM(v.cantidad) DESC) AS RankEstacional
                FROM dbo.FactVentas v
                JOIN dbo.DimProductos p ON v.ProductoKey = p.ProductoKey
                JOIN dbo.DimTiempo t ON v.TiempoKey = t.TiempoKey
                GROUP BY p.nombre,
                    CASE
                        WHEN t.Mes IN (12, 1, 2) THEN 'Invierno'
                        WHEN t.Mes IN (3, 4, 5) THEN 'Primavera'
                        WHEN t.Mes IN (6, 7, 8) THEN 'Verano'
                        ELSE 'Otoño'
                    END
            ),
            AnalisisRevelacion AS (
                SELECT ve.Producto, ve.Estacion, ve.CantidadEstacional, vg.RankGeneral, ve.RankEstacional,
                       (vg.RankGeneral - ve.RankEstacional) as MejoraDeRank
                FROM VentasEstacionales ve
                JOIN VentasGenerales vg ON ve.Producto = vg.Producto
                WHERE vg.RankGeneral > 10
            ),
            RankFinal AS (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY Estacion ORDER BY MejoraDeRank DESC) as RankMejora
                FROM AnalisisRevelacion
            )
            SELECT Producto, Estacion, CantidadEstacional, RankGeneral, RankEstacional, MejoraDeRank
            FROM RankFinal
            WHERE RankMejora <= 5
            ORDER BY Estacion, MejoraDeRank;
        """
        with st.spinner("Calculando reporte de productos revelación..."):
            df_revelacion = cargar_datos_sql(sql_query_revelacion)

        # --- Generación de Reporte (Revelación) ---
        if not df_revelacion.empty:
            for estacion, grupo in df_revelacion.groupby('Estacion'):
                with st.expander(f"Top 5 Productos Revelación para {estacion}"):
                    for index, row in grupo.iterrows():
                        st.write(
                            f"**{row['Producto']}**: Sube **{row['MejoraDeRank']}** puestos (del {row['RankGeneral']} al {row['RankEstacional']})")

            st.subheader("Visualización Comparativa")
            g = sns.catplot(
                data=df_revelacion, x='CantidadEstacional', y='Producto',
                col='Estacion', kind='bar', col_wrap=2, sharey=False,
                palette='magma', hue='Producto', legend=False
            )
            g.fig.suptitle('Top 5 Productos Revelación por Estación del Año', y=1.03, fontsize=16)
            g.set_axis_labels('Total de Unidades Vendidas en la Estación', 'Producto')
            g.set_titles("Estación: {col_name}")
            g.fig.tight_layout()
            st.pyplot(g.fig)
        else:
            st.error("No se pudieron cargar los datos de Productos Revelación.")


def show_page_baja_rotacion():
    """Página para el reporte de Baja Rotación (Entregable 5.3)."""
    st.title("Reporte: Productos de Baja Rotación (Candidatos a Liquidación)")
    st.markdown(
        "Este reporte identifica los 20 productos que se han vendido con **menor frecuencia** (en días distintos) durante los dos años de operación.")

    # --- Lógica de Carga de Datos ---
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
            DiasDeVenta ASC, UnidadesTotalesVendidas ASC;
    """
    with st.spinner("Buscando productos de baja rotación..."):
        df_baja_rotacion = cargar_datos_sql(sql_query_baja_rotacion)

    # --- Generación de Reporte ---
    if not df_baja_rotacion.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Reporte de Frecuencia y Volumen")
            st.dataframe(df_baja_rotacion)
            with st.expander("Análisis de Resultados"):
                st.write("""
                Los resultados muestran que los productos con menor frecuencia de venta se vendieron en 731 días.
                Esto significa que (casi) todos los productos tuvieron al menos una venta cada día de operación (731 días = 2 años).
                Por lo tanto, el gráfico de la derecha se enfoca en el **volumen total** para diferenciar a estos productos.
                """)

        with col2:
            st.subheader("Top 20 por Volumen (de los menos frecuentes)")
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.barplot(
                data=df_baja_rotacion,
                x='UnidadesTotalesVendidas',
                y='Producto',
                palette='autumn_r',
                hue='Producto',
                legend=False
            )
            ax.bar_label(ax.containers[0], fmt='{:,.0f}', padding=5, fontsize=9, color='black')
            ax.set_title('Top 20 Productos con Menor Frecuencia (por Volumen)', fontsize=16)
            ax.set_xlabel('Total de Unidades Vendidas en 2 Años', fontsize=12)
            ax.set_ylabel('Producto', fontsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            plt.xlim(right=df_baja_rotacion['UnidadesTotalesVendidas'].max() * 1.15)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("No se pudieron cargar los datos de Baja Rotación.")


def show_page_ia_kmeans():
    """Página para el modelo K-Means (Entregable 6)."""
    st.title("IA: Segmentación de Clientes con K-Means")
    st.markdown(
        "Este análisis utiliza Inteligencia Artificial (K-Means) para descubrir **5 perfiles de clientes** distintos basados en sus hábitos de compra (gasto por categoría).")

    @st.cache_data
    def entrenar_kmeans():
        sql_query = "SELECT TOP 100000 * FROM dbo.FactPerfilesDeCompra"
        df = cargar_datos_sql(sql_query)
        if df.empty:
            return None, None, None

        df_gastos = df.drop('id_ticket', axis=1)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_gastos)

        # Método del Codo
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_elbow.fit(df_scaled)
            inertia.append(kmeans_elbow.inertia_)

        k_optimo = 5
        kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
        # CORRECCIÓN 2: Usamos 'df_gastos' para asignar los labels, no 'df'
        df_gastos['cluster'] = kmeans.fit_predict(df_scaled)

        # Interpretar perfiles
        perfiles = df_gastos.groupby('cluster').mean(numeric_only=True)
        perfiles['tamaño_cluster'] = df_gastos['cluster'].value_counts()
        return perfiles, (K_range, inertia), df_gastos.columns[:-1]

    with st.spinner("Entrenando modelo K-Means y generando perfiles..."):
        perfiles, elbow_data, labels = entrenar_kmeans()

    if perfiles is not None:
        st.subheader("Visualización de los 5 Perfiles de Cliente")

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Cerramos el círculo

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        for i, row in perfiles.iterrows():
            data = row[labels].tolist()  # Usamos los labels correctos
            data += data[:1]  # Cerramos el círculo
            ax.plot(angles, data, linewidth=2, linestyle='solid',
                    label=f"Perfil {i} ({row['tamaño_cluster']:,} tickets)")
            ax.fill(angles, data, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='grey', size=8)
        ax.set_yticks([5, 10, 15, 20], ["$5", "$10", "$15", "$20"], color="grey", size=7)
        ax.set_ylim(0, max(perfiles.max()[:-1]) + 5)
        ax.set_title("Perfiles de Compra por Gasto Promedio en Categorías", size=20, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

        with st.expander("Ver Análisis Detallado de Perfiles y Método del Codo"):
            st.subheader("Análisis Detallado de Perfiles")
            st.markdown("""
            * **Perfil 0 (Fan de la Panadería/Desayuno):** Gasto más alto en `Bakery`.
            * **Perfil 1 (Comprador Esencial):** Gasto más bajo en casi todo. Es la compra rápida.
            * **Perfil 2 (Especialista en Marisco):** Gasto astronómico en `Seafood`.
            * **Perfil 3 (Planificador de Despensa):** Gasto más alto en `Canned Goods`.
            * **Perfil 4 (Comprador Semanal Equilibrado):** Gasto alto y balanceado en todas las categorías.
            """)
            st.dataframe(perfiles)

            st.subheader("Método del Codo (Elbow Method)")
            st.write(
                "Se usó este método para determinar el número óptimo de clusters. El 'codo' (punto de inflexión) se observa alrededor de k=4 o k=5, validando nuestra elección.")
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
            ax_elbow.plot(elbow_data[0], elbow_data[1], 'bx-')
            ax_elbow.set_xlabel('Número de Clusters (k)')
            ax_elbow.set_ylabel('Inertia')
            ax_elbow.set_title('Método del Codo para Encontrar k Óptimo')
            st.pyplot(fig_elbow)
    else:
        st.error("No se pudieron generar los perfiles de K-Means.")


def show_page_ia_svm():
    """Página para el modelo SVM (Entregable 6.1)."""
    st.title("IA: Evaluación del Clasificador de Clientes (SVM)")
    st.markdown(
        "Este reporte evalúa la precisión de nuestro modelo de IA (**Support Vector Machine**) para predecir a qué perfil pertenece un nuevo ticket de compra.")

    @st.cache_data
    def entrenar_svm():
        sql_query = "SELECT TOP 100000 * FROM dbo.FactPerfilesDeCompra"
        df = cargar_datos_sql(sql_query)
        if df.empty:
            return None, None, None, None

        df_gastos = df.drop('id_ticket', axis=1)

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_gastos)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['perfil_cliente'] = kmeans.fit_predict(df_scaled)

        X = df.drop(['id_ticket', 'perfil_cliente'], axis=1)
        y = df['perfil_cliente']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
        svm_model.fit(X_train_scaled, y_train)

        y_pred = svm_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        profile_names = [f'Perfil {i}' for i in range(5)]
        report_text = classification_report(y_test, y_pred, target_names=profile_names)
        report_dict = classification_report(y_test, y_pred, target_names=profile_names, output_dict=True)

        return accuracy, report_text, report_dict, profile_names

    with st.spinner("Entrenando y evaluando modelo SVM... (Esto puede tardar un momento)"):
        accuracy, report_text, report_dict, profile_names = entrenar_svm()

    if accuracy is not None:
        st.subheader(f"Precisión (Accuracy) General del Modelo: {accuracy * 100:.2f}%")
        st.success(
            f"El modelo es capaz de predecir correctamente el perfil del cliente en el **{accuracy * 100:.2f}%** de los casos nuevos.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Reporte de Clasificación Detallado")
            st.text(report_text)

            with st.expander("Haz clic aquí para entender estas métricas"):
                st.markdown("""
                * **Precision (Precisión):** De todas las veces que el modelo predijo un perfil (ej. "Perfil 0"), ¿qué porcentaje de veces acertó? *Un número alto significa que el modelo es confiable cuando hace una predicción.*

                * **Recall (Sensibilidad):** De todos los tickets que *realmente* eran de un perfil (ej. "Perfil 0"), ¿qué porcentaje de ellos logró encontrar el modelo? *Un número alto significa que el modelo es bueno para 'atrapar' a todos los miembros de un grupo.*

                * **F1-Score:** Es la media armónica (el balance) entre la Precisión y el Recall. Es la métrica más importante para juzgar el rendimiento general de un perfil. *Un número alto (cercano a 1.0) es excelente.*

                * **Support:** El número de instancias reales de ese perfil en el conjunto de prueba.

                * **Accuracy (Precisión General):** El porcentaje total de aciertos del modelo sobre todas las predicciones.
                """)

        with col2:
            st.subheader("Visualización del Reporte (Heatmap)")

            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df.loc[profile_names]
            report_df_visual = report_df[['precision', 'recall', 'f1-score']]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                report_df_visual,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                linewidths=0.5,
                ax=ax
            )
            ax.set_title('Mapa de Calor del Rendimiento del Modelo SVM', fontsize=16)
            ax.set_xlabel('Métricas de Evaluación', fontsize=12)
            ax.set_ylabel('Perfiles de Cliente', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("No se pudo entrenar o evaluar el modelo SVM.")


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    show_login_page()
    st.title("Por favor, inicie sesión")
    st.image("https://placehold.co/1200x300/22223B/E0E1DD?text=Bienvenido+al+Sistema+de+BI+de+'Poly'",
             use_column_width=True)
    st.write("Utilice el panel de la izquierda para ingresar sus credenciales.")
else:
    if st.session_state['role'] == 'admin':
        st.sidebar.title("Navegación de Reportes")
        pagina_seleccionada = st.sidebar.selectbox(
            "Seleccione un reporte:",
            [
                "--- Página Principal ---",
                "Patrones de Compra",
                "Estacionalidad",
                "Baja Rotación",
                "IA: Perfiles de Cliente (K-Means)",
                "IA: Clasificador (SVM)"
            ]
        )

        if pagina_seleccionada == "Patrones de Compra":
            show_page_patrones_compra()
        elif pagina_seleccionada == "Estacionalidad":
            show_page_estacionalidad()
        elif pagina_seleccionada == "Baja Rotación":
            show_page_baja_rotacion()
        elif pagina_seleccionada == "IA: Perfiles de Cliente (K-Means)":
            show_page_ia_kmeans()
        elif pagina_seleccionada == "IA: Clasificador (SVM)":
            show_page_ia_svm()
        else:
            show_page_home()

    else:
        show_page_home()

    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['logged_in'] = False
        st.session_state.pop('role', None)
        st.session_state.pop('username', None)
        st.rerun()
