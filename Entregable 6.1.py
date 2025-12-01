import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. CARGAR LOS DATOS Y ASIGNAR PERFILES CON K-MEANS
server_name = '.'
database_name = 'Poly'
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;"
sql_query = "SELECT TOP 100000 * FROM dbo.FactPerfilesDeCompra"

try:
    print("Cargando datos para el modelo de clasificación...")
    cnxn = pyodbc.connect(connection_string)
    df = pd.read_sql(sql_query, cnxn)
    cnxn.close()
    print("¡Datos cargados!")
except Exception as e:
    df = pd.DataFrame()
    print(f"Error al cargar los datos: {e}")

if not df.empty:
    df_gastos = df.drop('id_ticket', axis=1)

    print("Generando etiquetas de perfil de cliente con K-Means...")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_gastos)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['perfil_cliente'] = kmeans.fit_predict(df_scaled)

    # PREPARACIÓN PARA EL MODELO SUPERVISADO (SVM)

    X = df.drop(['id_ticket', 'perfil_cliente'], axis=1)
    y = df['perfil_cliente']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\nDatos listos: {len(X_train)} tickets para entrenamiento, {len(X_test)} para prueba.")

    # ENTRENAMIENTO DEL MODELO SVM
    print("Entrenando el modelo de Máquina de Soporte Vectorial (SVM)...")
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    print("¡Modelo entrenado exitosamente!")
    # EVALUACIÓN DEL MODELO (TEXTO)
    print("\n--- Evaluación del Rendimiento del Modelo ---")
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión del modelo (Accuracy): {accuracy * 100:.2f}%")

    # Preparamos los nombres de los perfiles para los reportes
    profile_names = [f'Perfil {i}' for i in range(5)]

    report_text = classification_report(y_test, y_pred, target_names=profile_names)
    print("\nReporte de Clasificación Detallado:")
    print(report_text)
    # VISUALIZACIÓN DEL RENDIMIENTO (HEATMAP)
    print("\nGenerando visualización del reporte (Heatmap)...")
    report_dict = classification_report(y_test, y_pred, target_names=profile_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.loc[profile_names]
    report_df_visual = report_df[['precision', 'recall', 'f1-score']]

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        report_df_visual,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        linewidths=0.5
    )

    plt.title('Mapa de Calor del Rendimiento del Modelo SVM por Perfil', fontsize=16)
    plt.xlabel('Métricas de Evaluación', fontsize=12)
    plt.ylabel('Perfiles de Cliente', fontsize=12)
    plt.tight_layout()
    plt.show()

