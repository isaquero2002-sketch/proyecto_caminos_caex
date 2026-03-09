
"""
Asignar consumo a pendientes usando KD-Tree en coordenadas proyectadas (metros).
- Proyecta TRAMOS y CAMIÓN 
- Exporta consumo medio por punto de tramo y por clase de pendiente.
- Incluye heatmap consolidado Consumo vs Pendiente por camión.

"""

import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import contextily as cx
from shapely.geometry import LineString



# ========================== CONFIGURACIÓN ==========================
BASE_PATH = Path(r"C:\Users\icquerov\OneDrive - Anglo American\Desktop\Proyecto_Caminos")
# cambia segun la base

# Entradas
DATOS_LIMPIOS_DIR = BASE_PATH / "outputs" / "datos_limpios"  
 # CSV limpios por camión
GEOMETRIA_DIR     = BASE_PATH / "pendientes_tramos"          
 # Carpeta que contiene las subcarpetas por tramo (ej: TR08)

# Salidas
OUT_DIR           = BASE_PATH / "outputs" / "analisis_cercania"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CSV camión
SEP               = ","              # separador de los CSV limpios
CHUNK_SIZE        = 200_000

# Umbral de asignación (m)
MAX_DIST_METERS   = 10.0

# Columnas de tramo 
COL_TRAMO_LAT   = "latitud"
COL_TRAMO_LON   = "longitud"
COL_TRAMO_PEND  = "pendiente_%"
COL_TRAMO_ALT   = "altitud_m"

# Columnas de camión 
COL_CAM_LAT     = "Latitud"
COL_CAM_LON     = "Longitud"
COL_CONSUMO     = "Fuel rate (L/h)"
COL_TIEMPO      = "Fecha AVL"
COL_CAM_VEL     = "Velocidad (Km/h)" # 
COL_CONSUMO_KM  = "Consumo (L/km)"   # 
COL_RPM         = "RPM"
COL_PEDAL       = "Pedal"
COL_FACTOR_CARGA= "F. de Carga"
COL_ESTADO_MOTOR= "estado_motor" 

# Filtros
FILTRO_CONSUMO_POSITIVO = True
FILTRO_GPS_CEROS        = False

# Mapas
GENERAR_HEATMAP    = True
GENERAR_MAPA       = True
#Control para el gráfico de dispersión
GENERAR_SCATTER_PENDIENTE_CONSUMO = True 
GENERAR_SCATTER_POR_TRAMO = True 
# ==================================================================


# ---------------------- Utilidades ----------------------
def build_tramo_kdtree(tramo_df, lat_col=COL_TRAMO_LAT, lon_col=COL_TRAMO_LON):
    """
    Proyecta (lat,lon) a metros (EPSG:32719, UTM 19S) y construye un KD-Tree.
    Retorna: tree, XY (Nx2 en metros), transformer (para proyectar camión).
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)  
    # (lon, lat) -> (x, y)
    x, y = transformer.transform(tramo_df[lon_col].values, tramo_df[lat_col].values)
    XY = np.column_stack([x, y])
    tree = cKDTree(XY)
    return tree, XY, transformer
# -------------------------------------------------------


def run_proximity_analysis_for_truck(nombre_camion, tramo_df_periodo, tree, transformer, periodo):
    """
    Asigna consumo a cada punto de tramo para un camión usando KD-Tree.
    Devuelve (out_pt, out_bin, all_matched_points).
    """
    print(f"\n--- Analizando Camión: {nombre_camion} para el período {periodo} ---")

    # Extraer año y mes del período para filtrar los datos del camión (ej: "2025-07")
    year, month = map(int, periodo.split('-'))

    CAMION_CSV = DATOS_LIMPIOS_DIR / nombre_camion / f"{nombre_camion}_consolidado_limpio.csv"
    if not CAMION_CSV.exists():
        print(f"  No se encontró: {CAMION_CSV}")
        return None, None, None

    # Diagnóstico rápido del archivo
    with open(CAMION_CSV, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip()
    print(f"Encabezados (1ra línea): {header_line[:200]}...")

    matched_points_chunks = []
    total_leido, total_asign = 0, 0

    reader = pd.read_csv(CAMION_CSV, sep=SEP, chunksize=CHUNK_SIZE,
                         low_memory=True, on_bad_lines='skip')
    for k, ch in enumerate(reader, start=1):
        ch.columns = ch.columns.str.strip()

        #  Asegurar que se lean todas las columnas necesarias, incluyendo estado_motor ---
        columnas_requeridas = [COL_CAM_LAT, COL_CAM_LON, COL_CONSUMO, COL_TIEMPO, COL_CAM_VEL, COL_RPM, COL_PEDAL, COL_FACTOR_CARGA, COL_ESTADO_MOTOR]
        # Filtrar solo las que realmente existen en el archivo para no dar error si alguna no está
        columnas_a_verificar = [c for c in columnas_requeridas if c in ch.columns]
        # Solo las críticas para error
        faltan = [c for c in [COL_CAM_LAT, COL_CAM_LON, COL_CONSUMO, COL_TIEMPO] if c not in ch.columns] 
        if faltan:
            print(f"  Faltan columnas clave en el chunk {k}: {faltan}. Saltando chunk.")
            continue

        ch = ch.copy()

        # Tipos
        ch[COL_CAM_LAT] = pd.to_numeric(ch[COL_CAM_LAT], errors='coerce')
        ch[COL_CAM_LON] = pd.to_numeric(ch[COL_CAM_LON], errors='coerce')
        ch[COL_CONSUMO] = pd.to_numeric(ch[COL_CONSUMO], errors='coerce')
        ch[COL_CAM_VEL] = pd.to_numeric(ch[COL_CAM_VEL], errors='coerce')
        ch[COL_TIEMPO]  = pd.to_datetime(ch[COL_TIEMPO], errors='coerce')

        # Filtrar por año/mes del período
        ch = ch[ch[COL_TIEMPO].dt.year == year]
        ch = ch[ch[COL_TIEMPO].dt.month == month]
        if ch.empty:
            continue

        total_leido += len(ch) # Contar puntos totales del período antes de filtros de calidad

        if k == 1:
            print("\n--- Diagnóstico CAMIÓN (Chunk 1) ---")
            print(f"lat: [{ch[COL_CAM_LAT].min():.6f}, {ch[COL_CAM_LAT].max():.6f}]")
            print(f"lon: [{ch[COL_CAM_LON].min():.6f}, {ch[COL_CAM_LON].max():.6f}]")
            pos = (ch[COL_CONSUMO] > 0).sum()
            print(f"registros consumo>0 (chunk1): {pos:,}")

        ch = ch.dropna(subset=[COL_CAM_LAT, COL_CAM_LON, COL_CONSUMO, COL_CAM_VEL])
        if FILTRO_GPS_CEROS:
            ch = ch[(ch[COL_CAM_LAT] != 0) & (ch[COL_CAM_LON] != 0)]
        if FILTRO_CONSUMO_POSITIVO:
            ch = ch[ch[COL_CONSUMO] > 0]
        if ch.empty:
            continue

        # Proyectar a UTM
        x_cam, y_cam = transformer.transform(ch[COL_CAM_LON].values, ch[COL_CAM_LAT].values)
        cam_XY = np.column_stack([x_cam, y_cam])

        # KD-Tree: vecino más cercano y distancia
        dist_m, idx_nn = tree.query(cam_XY, k=1, workers=-1)

        # Máscara por umbral
        mask = (dist_m <= MAX_DIST_METERS) & np.isfinite(dist_m)

        if mask.any():
            matched_chunk = ch[mask].copy()
            matched_chunk['idx_tramo'] = idx_nn[mask]
            matched_points_chunks.append(matched_chunk)

        total_asign += int(mask.sum())

        if k == 1:
            print(f"KDTree: chunk1 asignables ≤ {MAX_DIST_METERS} m: {mask.sum()} / {len(ch)}")
            if mask.any():
                p50, p95 = np.percentile(dist_m[mask], [50, 95])
                print(f"  · Dist NN p50/p95 (m): {p50:.1f} / {p95:.1f}")

        min_dist = float(np.nanmin(dist_m)) if np.isfinite(dist_m).any() else np.nan
        print(f"Chunk {k}: leídos {len(ch):,} | asignados {int(mask.sum()):,} | min dist (m): {min_dist if not np.isnan(min_dist) else 'N/A'}")

    if not matched_points_chunks:
        print("→ No se asignó ningún punto para este camión.")
        return None, None, None

    all_matched_points = pd.concat(matched_points_chunks, ignore_index=True)

    # Agregación por punto de tramo
    agg_dict = {
        COL_CONSUMO: ['mean', 'count'],
    }
    for col in [COL_RPM, COL_PEDAL, COL_FACTOR_CARGA]:
        if col in all_matched_points.columns:
            agg_dict[col] = 'mean'

    res_pt_agg = all_matched_points.groupby('idx_tramo').agg(agg_dict)
    res_pt_agg.columns = ['_'.join(col).strip() for col in res_pt_agg.columns.values]
    res_pt_agg = res_pt_agg.reset_index()

    # Unir con la geometría
    cols_join = ["idx_tramo", COL_TRAMO_LAT, COL_TRAMO_LON, "fase", "tramo", "direccion", COL_TRAMO_PEND]
    tramo_small = tramo_df_periodo[cols_join].copy()

    out_pt = tramo_small.merge(res_pt_agg, on="idx_tramo", how="left")

    # Guardar salidas por camión
    out_dir_cam = OUT_DIR / nombre_camion
    out_dir_cam.mkdir(parents=True, exist_ok=True)

    # CSV por punto
    por_punto_csv = out_dir_cam / f"{nombre_camion}_{periodo}_consumo_por_punto.csv"
    out_pt.to_csv(por_punto_csv, index=False)
    print(f" Exportado por punto: {por_punto_csv}")

    # Matched points crudos + info de tramo
    matched_points_path = out_dir_cam / f"{nombre_camion}_{periodo}_matched_points.xlsx"
    cols_a_unir = ['idx_tramo', 'tramo', 'direccion', COL_TRAMO_LAT, COL_TRAMO_LON]
    cols_a_unir.append(COL_TRAMO_PEND)

    out_matched = all_matched_points.merge(
        tramo_df_periodo[cols_a_unir],
        on='idx_tramo',
        how='left'
    )
    out_matched['camion'] = nombre_camion

    # Ahora se guarda el archivo
    out_matched.to_excel(matched_points_path, index=False, engine='openpyxl')
    print(f" Puntos asignados crudos guardados en: {matched_points_path}")

    # Por clase de pendiente
    out_bin = pd.DataFrame()
    has_consumo = f'{COL_CONSUMO}_count' in out_pt.columns and out_pt[f'{COL_CONSUMO}_count'].fillna(0).astype(int).gt(0).any()
    has_pend    = out_pt[COL_TRAMO_PEND].notna().any()

    if has_consumo and has_pend:
        pmin = float(np.nanmin(out_pt[COL_TRAMO_PEND].values))
        pmax = float(np.nanmax(out_pt[COL_TRAMO_PEND].values))
        nbins = 6
        bins = np.linspace(np.floor(pmin), np.ceil(pmax), nbins)
        if len(np.unique(bins)) < 2:
            bins = [pmin - 0.5, pmax + 0.5]

        out_pt["bin_pendiente"] = pd.cut(out_pt[COL_TRAMO_PEND], bins=bins, include_lowest=True)
        out_bin = (
            out_pt.loc[out_pt[f'{COL_CONSUMO}_count'] > 0]
                  .groupby("bin_pendiente")[f'{COL_CONSUMO}_mean']
                  .agg(["count", "median", "mean", "max"])
                  .reset_index()
        )
        por_bin_csv = out_dir_cam / f"{nombre_camion}_{periodo}_consumo_por_clase_pendiente.csv"
        out_bin.to_csv(por_bin_csv, index=False)
        print(f" Exportado por clase de pendiente: {por_bin_csv}")
    else:
        print("→ No hay consumo asignado o pendientes válidas para crear bins.")

    print(f"Total filas leídas: {total_leido:,} | Total asignadas (≤ {MAX_DIST_METERS} m): {total_asign:,}")
    return out_pt, out_bin, out_matched


def generar_mapa_puntos(tramo_df, consumo_df, out_dir):
    """
    Genera un mapa estático (imagen) con los tramos como puntos y los puntos de consumo
    coloreados por dirección (subida/bajada).

    Args:
        tramo_df (pd.DataFrame): DataFrame con la geometría de todos los tramos.
        consumo_df (pd.DataFrame): DataFrame con los puntos de consumo asignados.
        out_dir (Path): Directorio de salida para guardar el mapa.
    """
    print("\n→ Generando mapa estático con puntos de tramo y consumo por dirección...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 15))

    # Graficar puntos del tramo directamente
    # Proyectar para que contextily funcione correctamente
    gdf_tramos = gpd.GeoDataFrame(tramo_df, geometry=gpd.points_from_xy(tramo_df[COL_TRAMO_LON], tramo_df[COL_TRAMO_LAT]), crs="EPSG:4326").to_crs(epsg=3857)
    gdf_tramos.plot(ax=ax, color='gray', markersize=5, alpha=0.6, label='Geometría del Tramo')


    consumo_subida = consumo_df[consumo_df['direccion'] == 'Subida']
    consumo_bajada = consumo_df[consumo_df['direccion'] == 'Bajada']

    if not consumo_subida.empty:
        # Proyectar para que se superponga correctamente con el mapa base
        gdf_subida = gpd.GeoDataFrame(consumo_subida, geometry=gpd.points_from_xy(consumo_subida[COL_CAM_LON], consumo_subida[COL_CAM_LAT]), crs="EPSG:4326")
        gdf_subida.to_crs(epsg=3857).plot(ax=ax, color='red', markersize=10, alpha=0.5, label='Consumo (Subida)')

    if not consumo_bajada.empty:
        gdf_bajada = gpd.GeoDataFrame(consumo_bajada, geometry=gpd.points_from_xy(consumo_bajada[COL_CAM_LON], consumo_bajada[COL_CAM_LAT]), crs="EPSG:4326")
        gdf_bajada.to_crs(epsg=3857).plot(ax=ax, color='blue', markersize=10, alpha=0.5, label='Consumo (Bajada)')

    # Añadir mapa base
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

    ax.set_title('Mapa de Tramos y Consumo por Dirección', fontsize=16)
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.set_axis_off() # Ocultar los ejes de coordenadas proyectadas
    ax.legend(title='Leyenda', loc='upper left')

    map_path = out_dir / "mapa_puntos_tramos_consumo.png"
    plt.savefig(map_path, dpi=200, bbox_inches='tight')
    print(f" Mapa estático guardado en: {map_path}")

def generar_scatter_pendiente_consumo(consumo_df, out_dir):
    """
    Genera un gráfico de dispersión de Pendiente vs. Consumo.

    Args:
        consumo_df (pd.DataFrame): DataFrame con los puntos de consumo asignados,
                                   debe contener 'pendiente_%', 'Consumo (L/km)' y 'direccion'.
                                   También necesita la columna 'Pedal'.
        out_dir (Path): Directorio de salida para guardar el gráfico.
    """
    print("\n→ Generando gráfico de dispersión Pendiente vs. Consumo...")

    columnas_necesarias = [COL_TRAMO_PEND, COL_CONSUMO_KM, 'direccion', COL_PEDAL]
    if consumo_df.empty or any(col not in consumo_df.columns for col in columnas_necesarias):
        print(f"  No hay datos suficientes para generar el gráfico de dispersión (faltan columnas: {columnas_necesarias}).")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- NUEVA MODIFICACIÓN: Crear categorías basadas en dirección y pedal ---
    df_plot = consumo_df.copy()
    df_plot[COL_PEDAL] = pd.to_numeric(df_plot[COL_PEDAL], errors='coerce')

    conditions = [
        (df_plot['direccion'] == 'Subida') & (df_plot[COL_PEDAL] >= 80),
        (df_plot['direccion'] == 'Subida') & (df_plot[COL_PEDAL] < 80),
        (df_plot['direccion'] == 'Bajada')
    ]
    choices = ['Subida (Pedal > 80%)', 'Subida (Pedal < 80%)', 'Bajada']
    df_plot['categoria_conduccion'] = np.select(conditions, choices, default='Otro')

    # Definir una paleta de colores para las nuevas categorías
    palette = {
        'Subida (Pedal > 80%)': 'red',
        'Subida (Pedal < 80%)': 'orange',
        'Bajada': 'blue'
    }

    sns.scatterplot(
        data=df_plot,
        x=COL_TRAMO_PEND,
        y=COL_CONSUMO_KM, # <-- EJE Y AHORA ES L/km
        hue='categoria_conduccion', # Colorear por la nueva categoría
        palette=palette,
        alpha=0.3, # Puntos semitransparentes para ver la densidad
        s=15,      # Tamaño de los puntos
        ax=ax
    )

    ax.set_title('Dispersión de Consumo (L/km) vs. Pendiente por Tipo de Conducción', fontsize=16, weight='bold')
    ax.set_xlabel('Pendiente del Tramo (%)', fontsize=12)
    ax.set_ylabel(f'Consumo ({COL_CONSUMO_KM})', fontsize=12)

    ax.legend(title='Tipo de Conducción', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    scatter_path = out_dir / "scatter_consumo_vs_pendiente.png"
    # Usamos bbox_inches='tight' para asegurar que la leyenda externa se guarde correctamente.
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight') 
    print(f" Gráfico de dispersión guardado en: {scatter_path}")

def generar_scatter_por_tramo(consumo_df, out_dir):
    """
    Genera un gráfico de dispersión de Pendiente vs. Consumo (L/km) para cada tramo,
    coloreado por tipo de conducción (subida/bajada/pedal).
    """
    print("\n→ Generando gráficos de dispersión por tramo...")

    # Crear subdirectorio para estos gráficos
    scatters_dir = out_dir / "scatters_por_tramo"
    scatters_dir.mkdir(exist_ok=True)

    # Asegurarse de que las columnas necesarias existen
    columnas_necesarias = [COL_TRAMO_PEND, COL_CONSUMO_KM, 'direccion', COL_PEDAL, 'tramo']
    if consumo_df.empty or any(col not in consumo_df.columns for col in columnas_necesarias):
        print(f"  No hay datos suficientes para generar los gráficos por tramo (faltan columnas).")
        return

    # Crear la columna de categoría de conducción una sola vez
    df_plot = consumo_df.copy()
    df_plot[COL_PEDAL] = pd.to_numeric(df_plot[COL_PEDAL], errors='coerce')
    conditions = [
        (df_plot['direccion'] == 'Subida') & (df_plot[COL_PEDAL] >= 80),
        (df_plot['direccion'] == 'Subida') & (df_plot[COL_PEDAL] < 80),
        (df_plot['direccion'] == 'Bajada')
    ]
    choices = ['Subida (Pedal > 80%)', 'Subida (Pedal < 80%)', 'Bajada']
    df_plot['categoria_conduccion'] = np.select(conditions, choices, default='Otro')

    # Definir la paleta de colores
    palette = {
        'Subida (Pedal > 80%)': 'red',
        'Subida (Pedal < 80%)': 'orange',
        'Bajada': 'blue'
    }

    # Iterar por cada tramo y generar un gráfico
    for tramo_id, df_tramo in df_plot.groupby('tramo'):
        if df_tramo.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.scatterplot(
            data=df_tramo,
            x=COL_TRAMO_PEND,
            y=COL_CONSUMO_KM,
            hue='categoria_conduccion',
            palette=palette,
            alpha=0.5,
            s=20,
            ax=ax
        )

        ax.set_title(f'Tramo {tramo_id}: Consumo (L/km) vs. Pendiente por Tipo de Conducción', fontsize=16, weight='bold')
        ax.set_xlabel('Pendiente del Tramo (%)', fontsize=12)
        ax.set_ylabel(f'Consumo ({COL_CONSUMO_KM})', fontsize=12)
        ax.legend(title='Tipo de Conducción', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        scatter_path = scatters_dir / f"scatter_tramo_{tramo_id}.png"
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   Gráfico para tramo '{tramo_id}' guardado.")

def run_proximity_analysis(camiones_a_analizar, mes_filtro=None):
    # --- Cargar archivos de TRAMOS ---
    print(f"→ Buscando archivos de geometría recursivamente en: {GEOMETRIA_DIR}")
    all_tramo_files = []
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        all_tramo_files.extend(GEOMETRIA_DIR.rglob(ext))

    if not all_tramo_files:
        raise FileNotFoundError(f"No se encontraron archivos de geometría en: {GEOMETRIA_DIR}")

    tramo_dfs = []
    for file in all_tramo_files:
        print(f"  -> Leyendo archivo: {file.relative_to(BASE_PATH)}")
        df_leido = pd.read_excel(file, header=0) if file.suffix.startswith('.xls') else \
                   pd.read_csv(file, sep=None, engine='python', header=0, encoding='utf-8-sig')
        df_leido['__source_file'] = file.stem
        tramo_dfs.append(df_leido)

    tramo = pd.concat(tramo_dfs, ignore_index=True)

    # Normalizar nombres de columnas a minúsculas
    tramo.columns = [c.strip().lower() for c in tramo.columns]

    # Renombrar coordenadas
    col_map = {
        "y": COL_TRAMO_LAT, "lat": COL_TRAMO_LAT, "latitud": COL_TRAMO_LAT,
        "x": COL_TRAMO_LON, "lon": COL_TRAMO_LON, "lng": COL_TRAMO_LON, "longitud": COL_TRAMO_LON
    }
    tramo = tramo.rename(columns=col_map)

    # Renombrar altitud si viene como 'altitud'
    if "altitud" in tramo.columns and COL_TRAMO_ALT not in tramo.columns:
        tramo = tramo.rename(columns={"altitud": COL_TRAMO_ALT})

    if COL_TRAMO_LAT not in tramo.columns or COL_TRAMO_LON not in tramo.columns:
        raise ValueError("No se localizaron columnas de Latitud/Longitud en los archivos de tramos.")

    # Si viene 'nombre_feature', renombrar a 'nombre'
    if 'nombre_feature' in tramo.columns and 'nombre' not in tramo.columns:
        tramo = tramo.rename(columns={'nombre_feature': 'nombre'})

    if 'nombre' not in tramo.columns:
        print("  La columna 'nombre' no se encontró. Se usará '__source_file' como fallback.")
        tramo['nombre'] = tramo['__source_file']

    # Normalizar lat/lon: coma → punto y a número
    for c in [COL_TRAMO_LAT, COL_TRAMO_LON]:
        tramo[c] = tramo[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        tramo[c] = pd.to_numeric(tramo[c], errors="coerce")

    # Extraer período desde el nombre del archivo
    def periodo_desde_filename(path_str):
        name = Path(path_str).stem.lower()
        meses = {
            "enero": "01", "febrero": "02", "marzo": "03", "abril": "04", "mayo": "05", "junio": "06",
            "julio": "07", "agosto": "08", "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
        }
        for mes_nombre, mes_num in meses.items():
            if mes_nombre in name:
                match_year = re.search(r'(20\d{2})', name)
                año = match_year.group(1) if match_year else "2025"
                return f"{año}-{mes_num}"
        return "SIN_PERIODO"

    if 'periodo' not in tramo.columns:
        tramo['periodo'] = tramo['__source_file'].apply(periodo_desde_filename)

    tramo['periodo'] = tramo['periodo'].fillna("SIN_PERIODO")

    print("\nConteo de puntos de geometría por período:")
    print(tramo['periodo'].value_counts())

    # Extraer fase, tramo, dirección y pendiente desde 'nombre'
    extract_pattern = r'^(?P<Fase>[A-Z0-9]+)_(?P<Tramo>TR\d+)_(?P<Direccion>[BS])_(?P<Pendiente>-?\d+)%?$'
    extracted = tramo['nombre'].astype(str).str.upper().str.extract(extract_pattern)

    tramo['fase']      = extracted['Fase']
    tramo['tramo']     = extracted['Tramo']
    tramo['direccion'] = extracted['Direccion'].map({'S': 'Subida', 'B': 'Bajada'})

    tramo['tramo'] = tramo['tramo'].replace(['TR44', 'tr44'], 'TR42')

    tramo[COL_TRAMO_PEND] = pd.to_numeric(extracted['Pendiente'], errors='coerce')

    # Limpieza final
    tramo = tramo.dropna(subset=[COL_TRAMO_LAT, COL_TRAMO_LON, COL_TRAMO_PEND, 'fase', 'tramo', 'direccion']).reset_index(drop=True)

    print(f"TRAMOS WGS84 lat:[{tramo[COL_TRAMO_LAT].min():.6f},{tramo[COL_TRAMO_LAT].max():.6f}] "
          f"lon:[{tramo[COL_TRAMO_LON].min():.6f},{tramo[COL_TRAMO_LON].max():.6f}]")

    #  Guardar la geometría consolidada ANTES de iterar por períodos ---
    # Esto asegura que el archivo tenga todas las columnas necesarias para el script pasadas_por_tramo.py
    tramo.to_csv(OUT_DIR.parent / "geometria_consolidada_con_altitud.csv", index=False)
    print(f" Geometría consolidada guardada en: {OUT_DIR.parent / 'geometria_consolidada_con_altitud.csv'}")

    # --- FILTRAR POR MES SI SE SOLICITA ---
    if mes_filtro:
        # Asegurar que sea una lista para soportar múltiples meses
        if isinstance(mes_filtro, str):
            meses_lista = [mes_filtro]
        else:
            meses_lista = mes_filtro
            
        print(f"\n  Filtro activado: Procesando únicamente los meses: {', '.join(meses_lista)}")
        tramo = tramo[tramo['periodo'].isin(meses_lista)]
        if tramo.empty:
            print(f" No se encontraron geometrías para los meses seleccionados. Verifica el formato (YYYY-MM) o los archivos disponibles.")
            return

    all_results_binned = []
    all_results_by_point = []
    all_matched_points_consolidated = [] # Para el mapa

    # Iterar por período (mes)
    for periodo, tramo_periodo_df in tramo.groupby('periodo'):
        if periodo == 'SIN_PERIODO':
            print("  Saltando puntos de geometría sin período asignado.")
            continue

        print(f"\n\n{'='*20} PROCESANDO PERÍODO: {periodo} {'='*20}")
        tramo_periodo_df = tramo_periodo_df.copy()
        tramo_periodo_df["idx_tramo"] = np.arange(len(tramo_periodo_df))

        # KD-Tree para ese período
        tree, _, transformer = build_tramo_kdtree(tramo_periodo_df)

        for nombre_camion in camiones_a_analizar:
            out_pt, out_bin, df_matched = run_proximity_analysis_for_truck(
                nombre_camion, tramo_periodo_df, tree, transformer, periodo
            )

            if out_bin is not None and not out_bin.empty:
                out_bin['camion'] = nombre_camion
                out_bin['periodo'] = periodo
                all_results_binned.append(out_bin)

            if out_pt is not None and not out_pt.empty:
                out_pt['camion'] = nombre_camion
                out_pt['periodo'] = periodo
                all_results_by_point.append(out_pt)
            
            if df_matched is not None and not df_matched.empty:
                all_matched_points_consolidated.append(df_matched)


    # Heatmap consolidado
    if GENERAR_HEATMAP and all_results_by_point:
        print("\n→ Generando heatmap consolidado...")
        consolidated_points = pd.concat(all_results_by_point, ignore_index=True)
        consolidated_points['pendiente_redondeada'] = consolidated_points[COL_TRAMO_PEND].round(0).astype(int)

        heatmap_data = consolidated_points.pivot_table(
            index='pendiente_redondeada',
            columns='camion',
            values=f'{COL_CONSUMO}_mean',
            aggfunc='mean'
        )

        plt.figure(figsize=(max(8, len(camiones_a_analizar) * 2), 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=.5,
            cbar_kws={'label': 'Consumo Medio (L/h)'}
        )
        plt.title('Consumo Medio (L/h) vs. Pendiente por Camión')
        plt.xlabel('Camión')
        plt.ylabel('Pendiente (%)')
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        heatmap_path = OUT_DIR / "heatmap_consumo_vs_pendiente.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f" Heatmap guardado en: {heatmap_path}")

    # --- Generación de reportes consolidados y gráficos ---
    if all_matched_points_consolidated:
        consumo_consolidado = pd.concat(all_matched_points_consolidated, ignore_index=True)

        # 1. Resumen de asignación por tramo (con porcentajes)
        print("\n--- Resumen de asignación por tramo ---")
        if not consumo_consolidado.empty and 'tramo' in consumo_consolidado.columns:
            resumen_tramo = consumo_consolidado.groupby('tramo').size().reset_index(name='puntos_asignados')
            total_asignados = resumen_tramo['puntos_asignados'].sum()
            if total_asignados > 0:
                resumen_tramo['%_del_total'] = (resumen_tramo['puntos_asignados'] / total_asignados * 100).round(2)
            else:
                resumen_tramo['%_del_total'] = 0.0
            
            resumen_tramo = resumen_tramo.sort_values('puntos_asignados', ascending=False)
            
            resumen_path = OUT_DIR / "resumen_asignacion_por_tramo.csv"
            resumen_tramo.to_csv(resumen_path, index=False)
            print(f" Tabla resumen guardada en: {resumen_path}")
            print(resumen_tramo.head(15).to_string(index=False))

        # 2. Mapa de puntos consolidado
        if GENERAR_MAPA:
            generar_mapa_puntos(tramo, consumo_consolidado, OUT_DIR)

        # 3. Gráficos de dispersión
        if GENERAR_SCATTER_PENDIENTE_CONSUMO:
            generar_scatter_pendiente_consumo(consumo_consolidado, OUT_DIR)

        if GENERAR_SCATTER_POR_TRAMO:
            generar_scatter_por_tramo(consumo_consolidado, OUT_DIR)


if __name__ == "__main__":
    print("Iniciando análisis de cercanía de consumo a pendientes")

    lista_camiones = []
    mejores_camiones_file = OUT_DIR.parent / "mejores_camiones.txt"

    # Preguntar al usuario si quiere ingresar un camión específico
    val = input("Ingresa ID de camión (ej: CDH43, deja en blanco para usar la lista de mejores camiones): ").strip()

    if val:
        # Si el usuario ingresa un ID, usar ese camión
        lista_camiones = [v.strip() for v in val.split(",") if v.strip()]
    else:
        # Si el usuario deja en blanco, intentar leer la lista de mejores camiones
        if mejores_camiones_file.exists():
            with open(mejores_camiones_file, "r", encoding="utf-8", errors="ignore") as f:
                lista_camiones = [ln.strip() for ln in f if ln.strip()]
            if lista_camiones:
                print(f"Camiones desde archivo: {', '.join(lista_camiones)}")

    if not lista_camiones:
        print("No se seleccionaron camiones.")
            
    if lista_camiones:
        # Preguntar por mes o meses opcionales
        input_mes = input("Ingresa el mes o meses a procesar (YYYY-MM, separados por coma) o deja en blanco para procesar todos: ").strip()
        
        # Convertir entrada separada por comas en una lista
        meses_filtro = [m.strip() for m in input_mes.split(',')] if input_mes else None
        
        run_proximity_analysis(lista_camiones, mes_filtro=meses_filtro)
    else:
        print("No se seleccionaron camiones. Fin.")