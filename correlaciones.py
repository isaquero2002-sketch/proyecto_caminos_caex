"""
Analiza la variabilidad de los subtramos y genera matrices de correlación.

Funcionalidad:
1.  Lee el archivo 'coeficiente_variacion_subtramos.xlsx' que contiene las métricas
    de variabilidad (CV, skewness, IQR), promedio y mediana para cada subtramo.
2.  Para cada subtramo y cada parámetro, aplica un conjunto de reglas para decidir
    si el valor más representativo es el PROMEDIO o la MEDIANA.
3.  Agrupa los datos por 'camion' y 'tramo'.
4.  Para cada grupo, calcula una matriz de correlación sobre los valores representativos.
5.  Genera y guarda un mapa de calor (heatmap) para cada matriz de correlación.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- CONFIGURACIÓN ---
BASE_PATH = Path(r"C:\Users\icquerov\OneDrive - Anglo American\Desktop\Proyecto_Caminos")
OUTPUT_DIR = BASE_PATH / "outputs" / "analisis_correlacion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Archivo con los valores de promedio y mediana
DATOS_VALORES_FILE = BASE_PATH / "outputs" / "pasadas_mapas" / "subtramos_por_pasada.xlsx"

# 2. Archivo con las métricas para la decisión (CV, Skew)
DATOS_VARIABILIDAD_FILE = BASE_PATH / "outputs" / "analisis_variabilidad" / "coeficiente_variacion_subtramos.xlsx"

def generar_correlaciones_desde_variabilidad():
    """
    Función principal que carga los datos de variabilidad, aplica reglas para
    seleccionar el valor representativo y genera matrices de correlación por grupo.
    """
    print("--- Iniciando Análisis de Correlación desde Variabilidad de Subtramos ---")

    if not DATOS_VALORES_FILE.exists() or not DATOS_VARIABILIDAD_FILE.exists():
        print(f" Error: Faltan archivos de entrada. Asegúrate de que existan:")
        print(f"   - Archivo de valores: {DATOS_VALORES_FILE}")
        print(f"   - Archivo de variabilidad: {DATOS_VARIABILIDAD_FILE}")
        print("   Asegúrate de haber ejecutado 'pasadas_por_tramo.py' para generarlo.")
        return

    # 1. Cargar ambos archivos de datos
    try:
        print(f"Cargando valores (promedio/mediana) desde: {DATOS_VALORES_FILE}")
        df_valores = pd.read_excel(DATOS_VALORES_FILE)
        
        print(f"Cargando métricas de decisión (CV/skew) desde: {DATOS_VARIABILIDAD_FILE}")
        df_variabilidad = pd.read_excel(DATOS_VARIABILIDAD_FILE)
    except Exception as e:
        print(f" Error al leer el archivo Excel: {e}")
        return

    # Unir ambos DataFrames por las columnas clave para tener toda la información en un solo lugar
    id_cols = ['pasada_id', 'subtramo_id', 'camion', 'tramo']
    df_merged = pd.merge(df_valores, df_variabilidad, on=id_cols, how='inner', suffixes=('_val', '_var'))

    # El merge crea sufijos. Usamos la versión '_var' y la renombramos.
    # También aprovechamos para tomar las columnas de fecha de la misma fuente.
    if 'Direccion_Subtramo_var' in df_merged.columns:
        df_merged['Direccion_Subtramo'] = df_merged['Direccion_Subtramo_var']
    if 'fecha_inicio_var' in df_merged.columns:
        df_merged['fecha_inicio'] = df_merged['fecha_inicio_var']
    if 'fecha_fin_var' in df_merged.columns:
        df_merged['fecha_fin'] = df_merged['fecha_fin_var']

    # 2. Identificar los parámetros disponibles (buscando columnas de CV)
    parametros_base = [col.replace('CV_', '') for col in df_variabilidad.columns if col.startswith('CV_')]
    print(f"Se encontraron {len(parametros_base)} parámetros candidatos.")

    parametros_validos = []
    for param in parametros_base:
        required_cols = [
            f"{param}_prom",
            f"{param}_mediana",
            f"CV_{param}",
            f"Skew_{param}"
        ]
        if all(c in df_merged.columns for c in required_cols):
            parametros_validos.append(param)
    
    print(f"Se analizarán {len(parametros_validos)} parámetros que tienen todas las métricas requeridas (prom, mediana, CV, skew).")

    # 3. Crear el DataFrame consolidado aplicando las reglas
    columnas_iniciales = ['pasada_id', 'subtramo_id', 'camion', 'tramo', 'Direccion_Subtramo', 'n_puntos_val']
    if 'consumo_l_km' in df_merged.columns:
        columnas_iniciales.append('consumo_l_km')
    
    if 'desnivel_net_m' in df_merged.columns:
        columnas_iniciales.append('desnivel_net_m')
    if 'pendiente_local_pct' in df_merged.columns:
        columnas_iniciales.append('pendiente_local_pct')        
    if 'carga_ton' in df_merged.columns:
        columnas_iniciales.append('carga_ton')
    if 'consumo_l_kmton' in df_merged.columns:
        columnas_iniciales.append('consumo_l_kmton')
    
    if 'fecha_inicio' in df_merged.columns:
        columnas_iniciales.append('fecha_inicio')
    if 'fecha_fin' in df_merged.columns:
        columnas_iniciales.append('fecha_fin')
    if 'dist_2d_m' in df_merged.columns:
        columnas_iniciales.append('dist_2d_m')
    if 'dist_3d_m' in df_merged.columns:
        columnas_iniciales.append('dist_3d_m')
    if 'latitud' in df_merged.columns:
        columnas_iniciales.append('latitud')
    if 'longitud' in df_merged.columns:
        columnas_iniciales.append('longitud')

    df_consolidado = df_merged[columnas_iniciales].rename(columns={'n_puntos_val': 'n_puntos'}).copy()

    print("\n Aplicando reglas para consolidar valores (Promedio vs. Mediana)...")
    
    # Lista para almacenar el reporte de decisión
    reporte_decision = []

    for param in parametros_validos: # <-- Usar la lista de parámetros ya validados
        # Nombres de las columnas necesarias para este parámetro
        col_prom = f"{param}_prom"
        col_mediana = f"{param}_mediana"
        col_cv = f"CV_{param}"
        col_skew = f"Skew_{param}"

        # Verificar que todas las columnas necesarias existen
        # Rellenar NaNs en skew y cv para que las condiciones no fallen
        # Si skew es NaN, lo tratamos como 0 (simétrico), que favorece el promedio.
        # Si cv es NaN, lo tratamos como 0 (sin variabilidad), que favorece el promedio.        
        cv = df_merged[col_cv].fillna(0)
        skew = df_merged[col_skew].fillna(0)

        # Condición para usar el PROMEDIO:
        # Caso 1: n < 3
        # O
        # Caso 2: cv < 0.10 Y |skew| < 0.5
        condicion_usar_promedio = (
            (df_merged['n_puntos_val'] < 3) |
            ((cv < 0.10) & (abs(skew) < 0.5))
        )

        # Aplicar la lógica: si la condición es verdadera, usa el promedio; si no, usa la mediana.
        df_consolidado[param] = np.where(
            condicion_usar_promedio,
            df_merged[col_prom],
            df_merged[col_mediana]
        )
        
        # --- CÁLCULO DE ESTADÍSTICAS DE DECISIÓN ---
        n_total = len(df_merged)
        n_prom = condicion_usar_promedio.sum()
        n_med = n_total - n_prom
        
        # Verificar cuántos tienen desviación estándar válida (no nula)
        col_std = f"{param}_std"
        n_std_valida = df_merged[col_std].notna().sum() if col_std in df_merged.columns else 0

        reporte_decision.append({
            "Parametro": param,
            "Total_Subtramos": n_total,
            "Uso_Promedio": n_prom,
            "Pct_Promedio": round((n_prom / n_total * 100), 2) if n_total > 0 else 0,
            "Uso_Mediana": n_med,
            "Pct_Mediana": round((n_med / n_total * 100), 2) if n_total > 0 else 0,
            "Con_DesvEstandar_Valida": n_std_valida,
            "Pct_DesvEstandar_Valida": round((n_std_valida / n_total * 100), 2) if n_total > 0 else 0
        })

    # --- GUARDAR REPORTE DE DECISIÓN ---
    if reporte_decision:
        df_reporte = pd.DataFrame(reporte_decision)
        print("\n--- Resumen de Decisión Estadística (Promedio vs Mediana) ---")
        print(df_reporte[['Parametro', 'Pct_Promedio', 'Pct_Mediana', 'Pct_DesvEstandar_Valida']].to_string(index=False))
        
        reporte_path = OUTPUT_DIR / "resumen_decision_estadistica.xlsx"
        df_reporte.to_excel(reporte_path, index=False)
        print(f"  -> Reporte detallado guardado en: {reporte_path}")

    # Este archivo contiene una única columna por parámetro con el estadístico seleccionado.
    output_consolidado_file = OUTPUT_DIR / "datos_consolidados_para_correlacion.xlsx"
    df_consolidado.to_excel(output_consolidado_file, index=False, float_format="%.4f")
    print(f"\n Archivo con datos consolidados guardado en: {output_consolidado_file}")

    # 4. Iterar por cada grupo (camion, tramo) y generar la matriz de correlación
    print("\n Generando matrices de correlación por Camión, Tramo y Dirección...")
    
    # Filtrar solo las columnas de parámetros para la correlación
    columnas_para_corr = ['camion', 'tramo', 'Direccion_Subtramo', 'n_puntos'] + parametros_validos
    if 'consumo_l_km' in df_consolidado.columns:
        columnas_para_corr.append('consumo_l_km')
    if 'carga_ton' in df_consolidado.columns:
        columnas_para_corr.append('carga_ton')
    if 'consumo_l_kmton' in df_consolidado.columns:
        columnas_para_corr.append('consumo_l_kmton')

    df_para_corr = df_consolidado[columnas_para_corr].copy()

    # La lista ahora incluye los parámetros y 'consumo_l_km'
    columnas_numericas_a_verificar = parametros_validos + [c for c in ['consumo_l_km', 'carga_ton', 'consumo_l_kmton'] if c in df_para_corr.columns]
    
    for param in columnas_numericas_a_verificar:
        df_para_corr[param] = pd.to_numeric(df_para_corr[param], errors='coerce')

    for (camion, tramo, direccion), group_df in df_para_corr.groupby(['camion', 'tramo', 'Direccion_Subtramo']):
        if pd.isna(direccion): continue # Omitir grupos sin dirección definida
        print(f"  -> Procesando: Camión '{camion}', Tramo '{tramo}', Dirección '{direccion}' ({len(group_df)} subtramos)")

        # 1. Seleccionar solo las columnas numéricas.
        group_numeric_candidate = group_df.select_dtypes(include=np.number)
        
        # 2. Descartar columnas que estén completamente vacías (todo NaN).
        group_numeric_candidate = group_numeric_candidate.dropna(axis=1, how='all')
        
        # 3. Descartar columnas que no tengan variabilidad (son constantes).
        # La correlación no se puede calcular para una variable constante.
        group_numeric = group_numeric_candidate.loc[:, group_numeric_candidate.std() > 0]

        if group_numeric.shape[1] < 2:
            print("     -> No hay suficientes datos o variables para calcular la correlación. Saltando.")
            continue

        # Calcular la matriz de correlación
        corr_matrix = group_numeric.corr()

        # Guardar la matriz de correlación en Excel
        matrix_filename = f"matriz_corr_{camion}_{tramo}_{direccion}.xlsx"
        matrix_path = OUTPUT_DIR / matrix_filename
        corr_matrix.to_excel(matrix_path)
        print(f"     Matriz de correlación guardada en: {matrix_path}")

        # Generar el mapa de calor
        fig_size = max(12, len(corr_matrix.columns) * 0.6)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, vmin=-1, vmax=1)
        
        plt.title(f'Matriz de Correlación\nCamión: {camion} - Tramo: {tramo} - Dirección: {direccion}', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar el gráfico
        heatmap_filename = f"heatmap_corr_{camion}_{tramo}_{direccion}.png"
        heatmap_path = OUTPUT_DIR / heatmap_filename
        plt.savefig(heatmap_path, dpi=120)
        plt.close()
        print(f"     Mapa de calor guardado en: {heatmap_path}")

    print("\n--- Proceso de correlación completado. ---")

if __name__ == "__main__":
    generar_correlaciones_desde_variabilidad()
