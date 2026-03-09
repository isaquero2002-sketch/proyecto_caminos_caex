
import os, time, math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as cx

# =========================
# CONFIG
# =========================
BASE     = r"C:\Users\icquerov\OneDrive - Anglo American\Desktop\Proyecto_Caminos"
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CERCANIA_DIR    = os.path.join(OUT_DIR, "analisis_cercania")
TRAMOS_GEOMETRIA_CSV = Path(OUT_DIR) / "geometria_consolidada_con_altitud.csv"
GRAFICOS_ANALISIS_DIR = os.path.join(OUT_DIR, "analisis_pasadas_graficos")
MAPAS_PASADAS_DIR = os.path.join(OUT_DIR, "pasadas_mapas")

os.makedirs(MAPAS_PASADAS_DIR, exist_ok=True)

PASADAS_XLSX    = os.path.join(MAPAS_PASADAS_DIR, "pasadas_por_tramo.xlsx")
PUNTOS_INSTANTANEOS_XLSX = os.path.join(MAPAS_PASADAS_DIR, "puntos_instantaneos_por_pasada.xlsx")
PUNTOS_INSTANTANEOS_FEATHER = os.path.join(MAPAS_PASADAS_DIR, "puntos_instantaneos_por_pasada.feather")
SUBTRAMOS_PASADAS_XLSX = os.path.join(MAPAS_PASADAS_DIR, "subtramos_por_pasada.xlsx")
CORRELACION_XLSX = os.path.join(MAPAS_PASADAS_DIR, "pasadas_para_correlacion.xlsx")

MAX_GAP_MINUTES_NUEVA_PASADA = 6
PENDIENTE_UMBRAL_PCT = 1.0 # Umbral de pendiente (%) para clasificar Subida/Bajada/Plano
GENERAR_MAPAS_POR_PASADA = True
GENERAR_MAPA_CONSOLIDADO = True
MAX_MAPAS_A_GENERAR = 50

# --- CONFIGURACIÓN DE CARGAS ---
CARGAS_FILE = Path(BASE) / "camiones" / "carga_camiones" / "carga_tons.xlsx"
CARGAS_HEADER_ROW = 2

# ------------------ CONFIGURACIÓN DE CALIDAD ------------------
MIN_DIST_PASADA_M = 150
ALT_SMOOTH_WINDOW = 3
MAX_ABS_PENDIENTE_PCT = 20
TRATAR_PENDIENTE_OUTLIER_COMO_NAN = True

# =========================
# UTILS
# =========================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1; dlon = lo2 - lo1
    a = (np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*(np.sin(dlon/2)**2))
    return 2*R*np.arcsin(np.sqrt(a))

def resumen_largos_tramo(tramos_df):
    out = []
    group_cols = ["tramo", "direccion"]
    for (tramo_id, direccion), g in tramos_df.sort_values(group_cols + ["latitud", "longitud"]).groupby(group_cols):
        g = g.dropna(subset=["latitud", "longitud"]).reset_index(drop=True)
        if len(g) < 2:
            continue
        dist2d = 0.0
        dist3d = 0.0
        for i in range(len(g) - 1):
            p1 = (g.loc[i, "latitud"], g.loc[i, "longitud"])
            p2 = (g.loc[i + 1, "latitud"], g.loc[i + 1, "longitud"])
            d2 = haversine_m(p1[0], p1[1], p2[0], p2[1])
            dist2d += d2
            if 'altitud_m' in g.columns and pd.notna(g.loc[i, "altitud_m"]) and pd.notna(g.loc[i + 1, "altitud_m"]):
                dz = float(g.loc[i + 1, "altitud_m"] - g.loc[i, "altitud_m"])
                d3 = math.sqrt(d2**2 + dz**2)
            else:
                d3 = d2
            dist3d += d3
        out.append({
            "tramo": tramo_id,
            "Direccion_Tramo": direccion,
            "largo_geometrico_2d_m": dist2d,
            "largo_geometrico_3d_m": dist3d
        })
    return pd.DataFrame(out)

# =========================
# MAPS
# =========================
def plot_pasada_map(pasada_info, df_puntos_pasada, output_path):
    if df_puntos_pasada.empty:
        return

    gdf_puntos = gpd.GeoDataFrame(
        df_puntos_pasada,
        geometry=gpd.points_from_xy(df_puntos_pasada['Longitud'], df_puntos_pasada['Latitud']),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_puntos.plot(
        ax=ax,
        marker='o',
        column='Fuel rate (L/h)',
        cmap='YlOrRd',
        markersize=15,
        alpha=0.8,
        legend=True,
        legend_kwds={'label': "Consumo (L/h)", 'orientation': "horizontal"}
    )

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_title(f"Pasada en Tramo: {pasada_info['tramo']} por {pasada_info['camion']}\nInicio: {pasada_info['fecha_inicio']:%Y-%m-%d %H:%M}")
    ax.set_axis_off()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def plot_todas_las_pasadas_map(pasadas_df, all_points_df, tramo_geometria_df, output_path):
    if pasadas_df.empty or all_points_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    if tramo_geometria_df is not None and not tramo_geometria_df.empty:
        gdf_tramos = gpd.GeoDataFrame(
            tramo_geometria_df,
            geometry=gpd.points_from_xy(tramo_geometria_df['longitud'], tramo_geometria_df['latitud']),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        gdf_tramos.plot(ax=ax, color='gray', markersize=2, alpha=0.3, label='Geometría del Camino')

    num_pasadas = len(pasadas_df)
    cmap = plt.get_cmap('viridis', max(num_pasadas, 1))

    for i, pasada_info in pasadas_df.iterrows():
        pasada_id = pasada_info['pasada_id']
        df_puntos_pasada = all_points_df[all_points_df['pasada_id'] == pasada_id]
        if df_puntos_pasada.empty:
            continue

        gdf_puntos_pasada = gpd.GeoDataFrame(
            df_puntos_pasada,
            geometry=gpd.points_from_xy(df_puntos_pasada['Longitud'], df_puntos_pasada['Latitud']),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        gdf_puntos_pasada.plot(ax=ax, color=cmap(i / max(num_pasadas, 1)), markersize=5, alpha=0.6)

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_title(f"Mapa Consolidado de Todas las Pasadas ({num_pasadas} en total)")
    ax.set_axis_off()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =========================
# GRAPHS
# =========================
def generar_graficos_analisis(df, output_dir):
    print("\n Generando gráficos de análisis de pasadas...")
    os.makedirs(output_dir, exist_ok=True)

    df_clean = df[(df['consumo_l_km'].notna()) & (df['consumo_l_km'] > 0) & (df['consumo_l_km'] < 100)].copy()
    if df_clean.empty:
        print("  -> No hay datos suficientes para generar gráficos después de la limpieza.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    print("  -> Generando gráficos de Pendiente vs. Consumo por TRAMO...")
    for tramo_id in df_clean['tramo'].unique():
        df_single_tramo = df_clean[df_clean['tramo'] == tramo_id]
        if df_single_tramo.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        colores = {'Subida': 'red', 'Bajada': 'blue', 'Plano': 'green', 'Indeterminada': 'gray'}

        for direccion, datos in df_single_tramo.groupby('Direccion_Pasada'):
            ax.scatter(datos['pendiente_real_pct'], datos['consumo_l_km'],
                       color=colores.get(direccion, 'gray'), label=direccion, alpha=0.7)

        ax.set_xlabel('Pendiente Real de la Pasada (%)', fontsize=12)
        ax.set_ylabel('Consumo (L/km)', fontsize=12)
        ax.set_title(f'Tramo {tramo_id}: Dispersión de Pendiente vs. Consumo por Dirección', fontsize=14)
        ax.legend()
        ax.grid(True)

        filename = os.path.join(output_dir, f'dispersion_pendiente_consumo_tramo_{tramo_id}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("   Gráficos 'Pendiente vs. Consumo' por tramo guardados.")

    print("  -> Generando gráfico de dispersión de Consumo para todos los tramos...")
    orden_tramos = df_clean.groupby('tramo')['consumo_l_km'].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(max(15, len(orden_tramos) * 0.5), 8))
    sns.stripplot(x='tramo', y='consumo_l_km', data=df_clean, order=orden_tramos, jitter=0.2,
                  palette='viridis', ax=ax, alpha=0.7, size=4)
    sns.boxplot(x='tramo', y='consumo_l_km', data=df_clean, order=orden_tramos,
                color='white', ax=ax, width=0.3, linewidth=1, showfliers=False)

    plt.xlabel('Tramo', fontsize=12)
    plt.ylabel('Consumo (L/km)', fontsize=12)
    plt.title('Dispersión de Consumo (L/km) por Tramo (Todos los Tramos)', fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dispersion_consumo_todos_tramos.png"), dpi=150)
    plt.close()
    print("   Gráfico 'Dispersión de Consumo para todos los tramos' guardado.")

    print("  -> Generando gráfico de comparación de consumo promedio...")
    df_agg = df_clean.groupby(['tramo', 'Direccion_Pasada'])['consumo_l_km'].mean().reset_index()
    df_agg = df_agg[df_agg['Direccion_Pasada'].isin(['Subida', 'Bajada'])]

    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=df_agg, x='tramo', y='consumo_l_km', hue='Direccion_Pasada',
        palette={'Subida': 'orangered', 'Bajada': 'dodgerblue'}
    )
    plt.title('Consumo Promedio (L/km) por Tramo y Dirección', fontsize=16)
    plt.xlabel('Tramo', fontsize=12)
    plt.ylabel('Consumo Promedio (L/km)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Dirección')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparacion_consumo_promedio_tramos.png"), dpi=150)
    plt.close()
    print("   Gráfico 'Comparación de Consumo Promedio' guardado.")

def generar_graficos_eficiencia(df, output_dir):
    print("\n Generando gráficos de eficiencia por TRAMO (Subida/Bajada)...")

    output_dir_tramos = os.path.join(output_dir, "graficos_eficiencia_por_tramo")
    os.makedirs(output_dir_tramos, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    for tramo_id, df_tramo in df.groupby('tramo'):
        print(f"\n  -> Procesando Tramo: {tramo_id}")

        df_bajada = df_tramo[(df_tramo['Direccion_Pasada'] == 'Bajada') & (df_tramo['consumo_l_km'].notna())].copy()
        if not df_bajada.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.scatter(df_bajada['pendiente_real_pct'], df_bajada['consumo_l_km'], alpha=0.6, color='blue')
            ax.set_title(f'Tramo {tramo_id} - Bajada: Pendiente vs. Consumo (L/km)', fontsize=16)
            ax.set_xlabel('Pendiente Real (%)', fontsize=12)
            ax.set_ylabel('Consumo (L/km)', fontsize=12)
            ax.grid(True)
            filename = os.path.join(output_dir_tramos, f'eficiencia_bajada_tramo_{tramo_id}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("     Gráfico de bajada guardado.")
        else:
            print("    - No hay datos suficientes para el gráfico de bajada.")

        df_subida = df_tramo[(df_tramo['Direccion_Pasada'] == 'Subida') & (df_tramo['consumo_l_kmton'].notna())].copy()
        if not df_subida.empty:
            fig, ax1 = plt.subplots(figsize=(14, 8))
            ax1.set_xlabel('Pendiente Real (%)', fontsize=12)
            ax1.set_ylabel('Consumo Normalizado (L/km·ton)', fontsize=12)
            ax1.scatter(df_subida['pendiente_real_pct'], df_subida['consumo_l_kmton'], alpha=0.6)
            ax1.grid(True, linestyle='--', which='major', axis='y')
            plt.title(f'Tramo {tramo_id} - Subida: Consumo normalizado (L/km·ton)', fontsize=16)
            fig.tight_layout()
            filename = os.path.join(output_dir_tramos, f'eficiencia_subida_tramo_{tramo_id}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("     Gráfico de subida guardado.")
        else:
            print("    - No hay datos suficientes para el gráfico de subida (L/km·ton).")

def generar_grafico_variabilidad_distancia(df, output_dir):
    print("\n Generando gráfico de variabilidad de distancias 2D por tramo...")
    if df is None or df.empty or 'dist_2d_m' not in df.columns or 'tramo' not in df.columns:
        print("  -> No hay datos suficientes para generar el gráfico de variabilidad de distancias.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    orden_tramos = df.groupby('tramo')['dist_2d_m'].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(max(15, len(orden_tramos) * 0.6), 9))
    sns.boxplot(x='tramo', y='dist_2d_m', data=df, order=orden_tramos, ax=ax,
                color='skyblue', width=0.5, linewidth=1.5, showfliers=False)
    sns.stripplot(x='tramo', y='dist_2d_m', data=df, order=orden_tramos, ax=ax,
                  jitter=0.2, size=4, alpha=0.6, color='navy')

    ax.set_title('Variabilidad de la Distancia 2D de las Pasadas por Tramo', fontsize=16)
    ax.set_xlabel('Tramo', fontsize=12)
    ax.set_ylabel('Distancia 2D (m)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = os.path.join(output_dir, "variabilidad_distancia_2d_por_tramo.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"   Gráfico guardado en: {filename}")

# =========================
# CORE: PASADAS
# =========================
def construir_pasadas(df):
    df = df.dropna(subset=["tramo"]).copy()
    df = df.sort_values(["camion", "Fecha AVL"]).reset_index(drop=True)

    variables_a_agregar = [
        'RPM', 'Actual Speed (RPM)', 'F. de Carga', 'Percent Load At Current Speed (%)',
        'Engine Torque Mode ()', 'Actual Percent Torque (%)', 'Engine Demand - Percent Torque (%)',
        'Velocidad (Km/h)', 'Pedal', 'Accelerator position (%)', 'Potencia', 'Power (HP)',
        'Fuel rate (L/h)', 'EGT-AV (F)', 'Coolant temperature (F)', 'Fuel Temperature (F)',
        'Engine Oil Temperature (F)', 'Barometric Pressure (PSI)', 'Coolant Pressure (PSI)',
        'Injector Metering (PSI)', 'Pre-filter Oil Pressure (PSI)', 'IMP-LB (PSI)',
        'IMT-LBF (F)', 'Electrical potential (V)', 'IMT-LBR (F)', 'IMT-RBF (F)',
        'IMT-RBR (F)', 'IMP (PSI)', 'IMP-RB (MCRS) (PSI)', 'Oil Differential Pressure (PSI)',
        'Ecu temperature (F)', 'Rifle Oil Pressure (PSI)', 'Turbocharger Speed (RPM)',
        'EGT-RB (MCRS) (F)', 'EGT-LB (MCRS) (F)', 'Post Oil Filter (MCRS) (PSI)',
        'Engine Pre-filter Oil Pressure (Extended Range) (PSI)',
        'Engine Oil Filter Differential Pressure (Extended Range) (PSI)',
        'Engine Oil Filter Differential Pressure (Extended Range) (MCRS) (kPa)'
    ]
    for i in range(1, 17):
        variables_a_agregar.append(f'EGT-{i:02d} (F)')

    categorical_vars = [
        'estado_motor', 'Engine Operating State (bit)', 'Engine Oil Priming Pump Control (bit)',
        'Engine Controlled Shutdown Request (bit)', 'Engine Emergency (Immediate) Shutdown Indication (bit)',
        'SCR Thermal Management Active (bit)', 'Engine Oil Priming State (bit)'
    ]

    variables_a_agregar = [col for col in variables_a_agregar if col in df.columns]
    categorical_vars = [col for col in categorical_vars if col in df.columns]

    time_gap = (df['Fecha AVL'].diff() > pd.Timedelta(minutes=MAX_GAP_MINUTES_NUEVA_PASADA)).fillna(False)
    df["new_pass"] = (df["camion"] != df["camion"].shift()) | (df["tramo"] != df["tramo"].shift()) | time_gap
    df["pasada_id"] = df["new_pass"].cumsum()

    rows = []
    debug_counter = 0

    for pid, g in df.groupby("pasada_id"):
        if len(g) < 2:
            continue

        eq     = g["camion"].iloc[0]
        tramo  = g["tramo"].iloc[0]
        t0, t1 = g["Fecha AVL"].min(), g["Fecha AVL"].max()
        dur_s  = (t1 - t0).total_seconds()

        la = np.deg2rad(g["Latitud"].to_numpy())
        lo = np.deg2rad(g["Longitud"].to_numpy())
        dlat = np.diff(la)
        dlon = np.diff(lo)
        a = np.sin(dlat/2)**2 + np.cos(la[:-1])*np.cos(la[1:])*(np.sin(dlon/2)**2)
        d2d = 6371000 * 2*np.arcsin(np.sqrt(a))
        dist2d_m = float(np.nansum(d2d))

        if dist2d_m < MIN_DIST_PASADA_M:
            continue

        alt_limpia = None
        if "Altitude (m)" in g.columns and g["Altitude (m)"].notna().any():
            alt_limpia = (
                g["Altitude (m)"]
                .ffill().bfill()
                .rolling(window=ALT_SMOOTH_WINDOW, min_periods=1, center=True)
                .mean()
            )
            dz = np.diff(alt_limpia.to_numpy())
            d3d = np.sqrt(d2d**2 + dz**2)
            dist3d_m = float(np.nansum(d3d)) if len(d2d) > 0 else 0.0
            desnivel_net = float(alt_limpia.iloc[-1] - alt_limpia.iloc[0]) if len(alt_limpia) > 1 else 0.0
        else:
            dist3d_m = dist2d_m
            desnivel_net = np.nan

        # Calculamos pendiente real ANTES de clasificar la dirección
        pendiente_real = (desnivel_net / dist2d_m * 100) if (dist2d_m > 0 and pd.notna(desnivel_net)) else np.nan

        if pd.notna(pendiente_real):
            # Usamos porcentaje de pendiente en lugar de desnivel absoluto
            if pendiente_real > PENDIENTE_UMBRAL_PCT:
                direccion_pasada = 'Subida'
            elif pendiente_real < -PENDIENTE_UMBRAL_PCT:
                direccion_pasada = 'Bajada'
            else:
                direccion_pasada = 'Plano'
        else:
            direccion_pasada = g['direccion'].mode()[0] if ('direccion' in g.columns and not g['direccion'].mode().empty) else 'Indeterminada'

        if "Fuel rate (L/h)" in g.columns and "Fecha AVL" in g.columns and len(g) > 1:
            g_sorted = g.sort_values("Fecha AVL")
            delta_t_hours = g_sorted["Fecha AVL"].diff().dt.total_seconds() / 3600.0
            delta_t_hours = delta_t_hours.clip(lower=0).fillna(0)
            consumo_l = (g_sorted["Fuel rate (L/h)"] * delta_t_hours).sum()
            consumo_prom = float(g["Fuel rate (L/h)"].mean())
        else:
            consumo_l = np.nan
            consumo_prom = np.nan

        pendiente_es_outlier = False
        if pd.notna(pendiente_real) and abs(pendiente_real) > MAX_ABS_PENDIENTE_PCT:
            pendiente_es_outlier = True
            if TRATAR_PENDIENTE_OUTLIER_COMO_NAN:
                pendiente_real = np.nan
                direccion_pasada = 'Indeterminada'

        if debug_counter < 10:
            if alt_limpia is not None and len(alt_limpia) > 1:
                print(
                    f"[VALIDACIÓN] Pasada {pid} ({tramo}) "
                    f"AltIni={alt_limpia.iloc[0]:.1f} AltFin={alt_limpia.iloc[-1]:.1f} "
                    f"Desnivel={desnivel_net:.1f} Dist={dist2d_m:.1f} -> Pend={pendiente_real if pd.notna(pendiente_real) else 'NaN'} "
                    f"{'(OUTLIER)' if pendiente_es_outlier else ''}"
                )
            debug_counter += 1

        row_data = {
            "pasada_id": pid,
            "camion": eq,
            "tramo": tramo,
            "Direccion_Pasada": direccion_pasada,
            "fecha_inicio": t0,
            "fecha_fin": t1,
            "latitud": g["Latitud"].mean(),
            "longitud": g["Longitud"].mean(),
            "duracion_min": dur_s / 60.0,
            "dist_2d_m": dist2d_m,
            "dist_3d_m": dist3d_m,
            "desnivel_net_m": desnivel_net,
            "consumo_prom_l_h": consumo_prom,
            "consumo_total_l": consumo_l,
            "pendiente_real_pct": pendiente_real,
            "pendiente_outlier": int(pendiente_es_outlier),
            "n_puntos": len(g)
        }

        for col in variables_a_agregar:
            if g[col].notna().any():
                row_data[f"{col}_prom"] = g[col].mean()
                row_data[f"{col}_std"] = g[col].std()
                row_data[f"{col}_mediana"] = g[col].median()
            else:
                row_data[f"{col}_prom"] = np.nan
                row_data[f"{col}_std"] = np.nan
                row_data[f"{col}_mediana"] = np.nan

        if 'daño' in g.columns:
            row_data['daño_acumulado'] = g['daño'].sum()
            row_data['daño_maximo'] = g['daño'].max()
            row_data['eventos_daño'] = g['daño'].count()

        for col in categorical_vars:
            row_data[f"{col}_predominante"] = g[col].mode()[0] if not g[col].mode().empty else 'N/A'

        rows.append(row_data)

    df_pasadas = pd.DataFrame(rows)
    return df_pasadas, df

# =========================
# CORE: SUBTRAMOS
# =========================
def construir_subtramos(df_puntos_con_pasada_id):
    print("\n Construyendo subtramos homogéneos por pendiente...")

    df = df_puntos_con_pasada_id.sort_values(['pasada_id', 'Fecha AVL']).reset_index(drop=True)

    if 'Altitude (m)' in df.columns:
        df['altitud_limpia'] = df.groupby('pasada_id')['Altitude (m)'].transform(
            lambda x: x.ffill().bfill().rolling(window=ALT_SMOOTH_WINDOW, min_periods=1, center=True).mean()
        )
        df['dz'] = df.groupby('pasada_id')['altitud_limpia'].diff()
    else:
        df['altitud_limpia'] = np.nan
        df['dz'] = np.nan

    lat1 = np.radians(df['Latitud'])
    lon1 = np.radians(df['Longitud'])
    lat2 = np.radians(df.groupby('pasada_id')['Latitud'].shift())
    lon2 = np.radians(df.groupby('pasada_id')['Longitud'].shift())

    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df['d2d'] = 2 * 6371000 * np.arcsin(np.sqrt(a))

    variables_a_agregar = [
        'RPM', 'Actual Speed (RPM)', 'F. de Carga', 'Percent Load At Current Speed (%)',
        'Engine Torque Mode ()', 'Actual Percent Torque (%)', 'Engine Demand - Percent Torque (%)',
        'Velocidad (Km/h)', 'Pedal', 'Accelerator position (%)', 'Potencia', 'Power (HP)',
        'Fuel rate (L/h)', 'EGT-AV (F)', 'Coolant temperature (F)', 'Fuel Temperature (F)',
        'Engine Oil Temperature (F)', 'Barometric Pressure (PSI)', 'Coolant Pressure (PSI)',
        'Injector Metering (PSI)', 'Pre-filter Oil Pressure (PSI)', 'IMP-LB (PSI)',
        'IMT-LBF (F)', 'Electrical potential (V)', 'IMT-LBR (F)', 'IMT-RBF (F)',
        'IMT-RBR (F)', 'IMP (PSI)', 'IMP-RB (MCRS) (PSI)', 'Oil Differential Pressure (PSI)',
        'Ecu temperature (F)', 'Rifle Oil Pressure (PSI)', 'Turbocharger Speed (RPM)',
        'EGT-RB (MCRS) (F)', 'EGT-LB (MCRS) (F)', 'Post Oil Filter (MCRS) (PSI)',
        'Engine Pre-filter Oil Pressure (Extended Range) (PSI)',
        'Engine Oil Filter Differential Pressure (Extended Range) (PSI)',
        'Engine Oil Filter Differential Pressure (Extended Range) (MCRS) (kPa)'
    ]
    for i in range(1, 17):
        variables_a_agregar.append(f'EGT-{i:02d} (F)')
    variables_a_agregar = [col for col in variables_a_agregar if col in df.columns]

    subtramos_rows = []
    subtramo_counter = 0

    for pid, pasada_df in df.groupby('pasada_id'):

        if len(pasada_df) <= 3:
            subtramos_a_procesar = [(subtramo_counter, pasada_df)]
            subtramo_counter += 1
        else:
            pasada_df = pasada_df.copy()
            pasada_df['direccion_pendiente'] = np.sign(pasada_df['dz']).fillna(0)

            break_by_slope = (
                (pasada_df['direccion_pendiente'] != pasada_df['direccion_pendiente'].shift()) &
                (pasada_df['direccion_pendiente'] != 0) &
                (pasada_df['direccion_pendiente'].shift() != 0)
            ).fillna(False)

            temp_group = break_by_slope.cumsum()
            dist_acum_en_grupo = pasada_df.groupby(temp_group)['d2d'].cumsum()
            break_by_dist = dist_acum_en_grupo > 10

            pasada_df['subtramo_local_id'] = (break_by_slope | break_by_dist.shift(fill_value=False)).cumsum()

            subtramos_a_procesar = []
            for _, subtramo_g in pasada_df.groupby('subtramo_local_id'):
                subtramos_a_procesar.append((subtramo_counter, subtramo_g))
                subtramo_counter += 1

        for sid, g in subtramos_a_procesar:
            if len(g) < 2:
                continue

            t0, t1 = g['Fecha AVL'].min(), g['Fecha AVL'].max()
            dist2d_m = float(g['d2d'].sum())

            if not (1.0 < dist2d_m < 5000.0):
                continue

            desnivel_net = float(g['altitud_limpia'].iloc[-1] - g['altitud_limpia'].iloc[0]) if g['altitud_limpia'].notna().any() else 0.0

            d3d = np.sqrt(g['d2d']**2 + g['dz']**2) if g['dz'].notna().any() else g['d2d']
            dist3d_m = float(d3d.sum())

            consumo_l = 0.0
            if "Fuel rate (L/h)" in g.columns:
                delta_t_hours = g["Fecha AVL"].diff().dt.total_seconds().fillna(0) / 3600.0
                consumo_l = float((g["Fuel rate (L/h)"] * delta_t_hours).sum())

            pendiente_local_pct = (desnivel_net / dist2d_m * 100) if dist2d_m > 0 else 0.0

            # clamp local opcional (mismo umbral)
            if abs(pendiente_local_pct) > MAX_ABS_PENDIENTE_PCT:
                # no lo botamos; lo marcamos
                pendiente_local_outlier = 1
                if TRATAR_PENDIENTE_OUTLIER_COMO_NAN:
                    pendiente_local_pct = np.nan
            else:
                pendiente_local_outlier = 0

            if pd.isna(pendiente_local_pct):
                direccion = 'Indeterminada'
            elif pendiente_local_pct > 1:
                direccion = 'Subida'
            elif pendiente_local_pct < -1:
                direccion = 'Bajada'
            else:
                direccion = 'Plano'

            row_data = {
                "pasada_id": pid,
                "subtramo_id": sid,
                "camion": g["camion"].iloc[0],
                "tramo": g["tramo"].iloc[0],
                "Direccion_Subtramo": direccion,
                "fecha_inicio": t0,
                "fecha_fin": t1,
                "latitud": g["Latitud"].mean(),
                "longitud": g["Longitud"].mean(),
                "duracion_min": (t1 - t0).total_seconds() / 60.0,
                "dist_2d_m": dist2d_m,
                "dist_3d_m": dist3d_m,
                "desnivel_net_m": desnivel_net,
                "pendiente_local_pct": pendiente_local_pct,
                "pendiente_local_outlier": int(pendiente_local_outlier),
                "consumo_total_l": consumo_l,
                "n_puntos": len(g)
            }

            for col in variables_a_agregar:
                if g[col].notna().any():
                    row_data[f"{col}_prom"] = g[col].mean()
                    row_data[f"{col}_std"] = g[col].std()
                    row_data[f"{col}_mediana"] = g[col].median()
                    row_data[f"{col}_skew"] = g[col].skew()
                    q1 = g[col].quantile(0.25)
                    q3 = g[col].quantile(0.75)
                    row_data[f"{col}_iqr"] = q3 - q1
                else:
                    row_data[f"{col}_prom"] = np.nan
                    row_data[f"{col}_std"] = np.nan
                    row_data[f"{col}_mediana"] = np.nan
                    row_data[f"{col}_skew"] = np.nan
                    row_data[f"{col}_iqr"] = np.nan

            if 'daño' in g.columns:
                row_data['daño_acumulado'] = g['daño'].sum()
                row_data['daño_maximo'] = g['daño'].max()
                row_data['eventos_daño'] = g['daño'].count()

            subtramos_rows.append(row_data)

    df_subtramos = pd.DataFrame(subtramos_rows)
    print(f"  -> Se construyeron {len(df_subtramos):,} subtramos a partir de las pasadas.")
    return df_subtramos

def analizar_distribucion_subtramos(df_puntos_con_pasada_id, output_dir):
    print("\n Analizando distribución de subtramos (pre-filtrado)...")
    df = df_puntos_con_pasada_id.sort_values(['pasada_id', 'Fecha AVL']).reset_index(drop=True)

    df['dz'] = df.groupby('pasada_id')['Altitude (m)'].transform(lambda x: x.ffill().bfill().diff())
    lat1 = np.radians(df['Latitud']); lon1 = np.radians(df['Longitud'])
    lat2 = np.radians(df.groupby('pasada_id')['Latitud'].shift()); lon2 = np.radians(df.groupby('pasada_id')['Longitud'].shift())
    dlat = lat1 - lat2; dlon = lon1 - lon2
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df['d2d'] = 2 * 6371000 * np.arcsin(np.sqrt(a))
    df['direccion_pendiente'] = np.sign(df['dz']).fillna(0)

    break_by_slope = df.groupby('pasada_id')['direccion_pendiente'].transform(
        lambda x: (x != x.shift()) & (x != 0) & (x.shift() != 0)
    ).fillna(False)

    temp_group = break_by_slope.cumsum()
    dist_acum_en_grupo = df.groupby(temp_group)['d2d'].cumsum()
    break_by_dist = dist_acum_en_grupo > 10
    df['subtramo_id_potencial'] = (break_by_slope | break_by_dist.shift(fill_value=False)).cumsum()

    distancias_subtramos = df.groupby(['pasada_id', 'subtramo_id_potencial'])['d2d'].sum().reset_index()
    distancias_subtramos = distancias_subtramos.rename(columns={'d2d': 'longitud_subtramo_m'})

    limite_inferior = 1.0
    limite_superior = 5000.0

    estado_valido = f'Válido ({limite_inferior}m < d < {limite_superior}m)'
    estado_corto = f'Descartado (<= {limite_inferior}m)'
    estado_largo = f'Descartado (>= {limite_superior}m)'

    distancias_subtramos['estado'] = estado_valido
    distancias_subtramos.loc[distancias_subtramos['longitud_subtramo_m'] <= limite_inferior, 'estado'] = estado_corto
    distancias_subtramos.loc[distancias_subtramos['longitud_subtramo_m'] >= limite_superior, 'estado'] = estado_largo

    reporte_path = os.path.join(output_dir, "distribucion_subtramos_descartados.xlsx")
    distancias_subtramos.to_excel(reporte_path, index=False)
    print(f"  -> Reporte guardado en: {reporte_path}")
    print(f"  -> Resumen:\n{distancias_subtramos['estado'].value_counts(dropna=False)}")

    plt.figure(figsize=(12, 7))
    palette = {estado_valido: 'green', estado_corto: 'red', estado_largo: 'orange'}
    sns.histplot(data=distancias_subtramos, x='longitud_subtramo_m', hue='estado', multiple='stack', bins=120, palette=palette)
    plt.title('Histograma de Longitudes de Subtramos Potenciales', fontsize=16)
    plt.xlabel('Longitud del Subtramo (m)', fontsize=12)
    plt.ylabel('Cantidad', fontsize=12)
    plt.xlim(0, 200)  # si quieres ver más, sube a 1000
    plt.grid(True, linestyle='--', alpha=0.6)
    hist_path = os.path.join(output_dir, "histograma_longitud_subtramos.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"  -> Histograma guardado en: {hist_path}")

# =========================
# reportes
# =========================
def enriquecer_y_guardar_reportes(pasadas_df, subtramos_df):
    print("\n Enriqueciendo reportes y asignando cargas...")

    tramos_geometria = pd.read_csv(TRAMOS_GEOMETRIA_CSV) if TRAMOS_GEOMETRIA_CSV and TRAMOS_GEOMETRIA_CSV.exists() else None

    cargas_df = None
    if CARGAS_FILE.exists():
        try:
            cargas_df = pd.read_excel(CARGAS_FILE, header=CARGAS_HEADER_ROW)
            colmap = {"Truck": "Truck", "Fecha": "FullTimestamp", "Promedio de Tons": "PromedioTons"}
            for original_col in colmap.keys():
                if original_col not in cargas_df.columns:
                    raise ValueError(f"La columna '{original_col}' no se encontró en '{CARGAS_FILE.name}'.")

            cargas_df = cargas_df.rename(columns=colmap)
            cargas_df['FullTimestamp'] = pd.to_datetime(cargas_df['FullTimestamp'], errors='coerce')
            cargas_df = cargas_df.dropna(subset=['Truck', 'FullTimestamp', 'PromedioTons']).sort_values(['Truck', 'FullTimestamp'])
            print("  -> Archivo de cargas procesado correctamente.")
        except Exception as e:
            print(f"  -> Error al procesar cargas: {e}")
            cargas_df = None
    else:
        print("  -> Archivo de cargas no encontrado. No se calculará L/km·ton.")

    def asignar_carga(row):
        if cargas_df is None:
            return 0.0
        cargas_camion = cargas_df[cargas_df['Truck'] == row['camion']]
        if cargas_camion.empty:
            return 0.0
        mask = cargas_camion['FullTimestamp'] <= row['fecha_inicio']
        return float(cargas_camion.loc[mask, 'PromedioTons'].iloc[-1]) if mask.any() else 0.0

    def calcular_metricas_eficiencia(df_in):
        if df_in is None or df_in.empty:
            return df_in
        df_copy = df_in.copy()
        dist_km = df_copy['dist_3d_m'] / 1000.0
        df_copy['consumo_l_km'] = np.divide(df_copy['consumo_total_l'], dist_km.replace(0, np.nan))
        df_copy['carga_ton'] = df_copy.apply(asignar_carga, axis=1)
        mask_calculo = (df_copy['carga_ton'] > 0) & df_copy['consumo_l_km'].notna()
        df_copy.loc[mask_calculo, 'consumo_l_kmton'] = df_copy.loc[mask_calculo, 'consumo_l_km'] / df_copy.loc[mask_calculo, 'carga_ton']
        return df_copy

    reporte_final_pasadas = pasadas_df.copy()
    if tramos_geometria is not None:
        dist_tramos_df = resumen_largos_tramo(tramos_geometria)
        reporte_final_pasadas = pd.merge(
            reporte_final_pasadas, dist_tramos_df,
            left_on=["tramo", "Direccion_Pasada"],
            right_on=["tramo", "Direccion_Tramo"],
            how="left"
        )

    reporte_final_pasadas = calcular_metricas_eficiencia(reporte_final_pasadas)
    reporte_final_pasadas.to_excel(PASADAS_XLSX, index=False)
    print(f"  -> Reporte pasadas guardado en: {PASADAS_XLSX}")

    reporte_final_subtramos = calcular_metricas_eficiencia(subtramos_df) if subtramos_df is not None else None
    if reporte_final_subtramos is not None and not reporte_final_subtramos.empty:
        reporte_final_subtramos.to_excel(SUBTRAMOS_PASADAS_XLSX, index=False)
        print(f"  -> Reporte subtramos guardado en: {SUBTRAMOS_PASADAS_XLSX}")

    return reporte_final_pasadas, reporte_final_subtramos, tramos_geometria

# =========================
# SUBTRAMOS graficos 
# =========================
def generar_graficos_subtramos(df_subtramos, output_dir):
    if df_subtramos is None or df_subtramos.empty:
        print("\n No hay datos de subtramos para generar gráficos.")
        return

    print("\n Generando gráficos de análisis para SUBTRAMOS (por Tramo y por Camión)...")
    output_dir_subtramos = os.path.join(output_dir, "graficos_subtramos")
    os.makedirs(output_dir_subtramos, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    df_original = df_subtramos.copy()

    df_clean = df_subtramos[
        (df_subtramos['consumo_l_km'].notna()) & (df_subtramos['consumo_l_km'] > 0) & (df_subtramos['consumo_l_km'] < 100) &
        (df_subtramos['pendiente_local_pct'].notna()) & (df_subtramos['pendiente_local_pct'].abs() < MAX_ABS_PENDIENTE_PCT)
    ].copy()

    if df_clean.empty:
        print("  -> No hay datos de subtramos suficientes para generar gráficos después de la limpieza.")
        return

    print("  -> Generando gráficos por combinación de TRAMO y CAMIÓN...")
    for (tramo_id, camion_id), _ in df_original.groupby(['tramo', 'camion']):
        for direccion in ['Subida', 'Bajada']:

            df_lkm = df_clean[
                (df_clean['tramo'] == tramo_id) &
                (df_clean['camion'] == camion_id) &
                (df_clean['Direccion_Subtramo'] == direccion)
            ]

            if not df_lkm.empty:
                fig, ax = plt.subplots(figsize=(12, 7))
                color = 'red' if direccion == 'Subida' else 'blue'
                ax.scatter(df_lkm['pendiente_local_pct'], df_lkm['consumo_l_km'], alpha=0.5, color=color, s=10)
                ax.set_title(f'Subtramos {direccion}: Tramo {tramo_id} - Camión {camion_id}\nPendiente vs. Consumo (L/km)', fontsize=16)
                ax.set_xlabel('Pendiente Local (%)', fontsize=12)
                ax.set_ylabel('Consumo (L/km)', fontsize=12)
                ax.grid(True)
                filename = os.path.join(output_dir_subtramos, f'subtramo_{tramo_id}_camion_{camion_id}_{direccion.lower()}_lkm.png')
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)

            df_lkmton_filtered = df_original[
                (df_original['tramo'] == tramo_id) &
                (df_original['camion'] == camion_id) &
                (df_original['Direccion_Subtramo'] == direccion) &
                (df_original['consumo_l_kmton'].notna()) & (df_original['consumo_l_kmton'] > 0) &
                (df_original['pendiente_local_pct'].notna()) & (df_original['pendiente_local_pct'].abs() < MAX_ABS_PENDIENTE_PCT)
            ].copy()

            if not df_lkmton_filtered.empty:
                fig, ax = plt.subplots(figsize=(12, 7))
                scatter = ax.scatter(
                    df_lkmton_filtered['pendiente_local_pct'],
                    df_lkmton_filtered['consumo_l_kmton'],
                    c=df_lkmton_filtered['carga_ton'],
                    cmap='viridis',
                    alpha=0.6, s=15
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Carga (ton)')
                ax.set_title(f'Subtramos {direccion}: Tramo {tramo_id} - Camión {camion_id}\nPendiente vs. Consumo Normalizado (L/km·ton)', fontsize=16)
                ax.set_xlabel('Pendiente Local (%)', fontsize=12)
                ax.set_ylabel('Consumo Normalizado (L/km·ton)', fontsize=12)
                ax.grid(True)
                filename = os.path.join(output_dir_subtramos, f'subtramo_{tramo_id}_camion_{camion_id}_{direccion.lower()}_lkmton.png')
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)

    print("   Gráficos por combinación de tramo y camión finalizados.")

def calcular_y_guardar_cv_subtramos(df_subtramos, base_output_dir):
    if df_subtramos is None or df_subtramos.empty:
        print("\n No hay datos de subtramos para calcular CV.")
        return

    print("\n Calculando Coeficiente de Variación (CV) para cada subtramo...")
    output_dir = os.path.join(base_output_dir, "analisis_variabilidad")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "coeficiente_variacion_subtramos.xlsx")

    id_cols = ['pasada_id', 'subtramo_id', 'camion', 'tramo', 'Direccion_Subtramo', 'fecha_inicio', 'fecha_fin', 'n_puntos']
    id_cols_existentes = [col for col in id_cols if col in df_subtramos.columns]
    df_resultado = df_subtramos[id_cols_existentes].copy()

    parametros_prom = [col for col in df_subtramos.columns if col.endswith('_prom')]
    print(f"  -> Se procesarán CV, Skew e IQR para {len(parametros_prom)} parámetros.")

    nuevas_metricas = {}
    for col_prom in parametros_prom:
        base_name = col_prom.replace('_prom', '')
        col_std = f"{base_name}_std"
        col_skew = f"{base_name}_skew"
        col_iqr = f"{base_name}_iqr"
        col_cv = f"CV_{base_name}"

        if col_std in df_subtramos.columns:
            promedio = df_subtramos[col_prom]
            desv_est = df_subtramos[col_std]
            nuevas_metricas[col_cv] = np.divide(desv_est, promedio.replace(0, np.nan))

        if col_skew in df_subtramos.columns:
            nuevas_metricas[f"Skew_{base_name}"] = df_subtramos[col_skew]
        if col_iqr in df_subtramos.columns:
            nuevas_metricas[f"IQR_{base_name}"] = df_subtramos[col_iqr]

    df_resultado = pd.concat([df_resultado, pd.DataFrame(nuevas_metricas)], axis=1)

    df_resultado.to_excel(output_file, index=False, float_format="%.4f")
    print(f"  -> Reporte CV guardado en: {output_file}")

# =========================
# MAIN
# =========================
def run_pasadas_analysis():
    print("Buscando datos de puntos asignados desde 'analizar_cercania'...")
    matched_files = list(Path(CERCANIA_DIR).rglob("*_matched_points.xlsx"))
    if not matched_files:
        raise FileNotFoundError(f"No se encontraron '*_matched_points.xlsx' en {CERCANIA_DIR}. Ejecuta 'analizar_cercania.py' primero.")

    all_matched_points = pd.concat([pd.read_excel(f) for f in matched_files], ignore_index=True)
    all_matched_points['Fecha AVL'] = pd.to_datetime(all_matched_points['Fecha AVL'], errors='coerce')
    print(f"  -> Total de puntos asignados cargados: {len(all_matched_points):,}")

    print("\n[1/4] Construyendo pasadas por tramo...")
    pasadas_df, all_matched_points_con_pasada_id = construir_pasadas(all_matched_points)

    if pasadas_df.empty:
        print("No se construyó ninguna pasada. El script terminará.")
        return

    # --- ESTADÍSTICAS DE REGISTROS (Asociados vs Descartados) ---
    total_registros = len(all_matched_points_con_pasada_id)
    registros_en_viajes = pasadas_df['n_puntos'].sum()
    registros_descartados = total_registros - registros_en_viajes
    pct_asociados = (registros_en_viajes / total_registros * 100) if total_registros > 0 else 0
    pct_descartados = (registros_descartados / total_registros * 100) if total_registros > 0 else 0

    print(f"\n--- Estadísticas de Registros en Viajes ---")
    print(f"Total registros procesados (asignados a tramos): {total_registros:,}")
    print(f"Registros asociados a viajes válidos: {registros_en_viajes:,} ({pct_asociados:.2f}%)")
    print(f"Registros descartados (viajes cortos/ruido): {registros_descartados:,} ({pct_descartados:.2f}%)")

    print(f"  -> Se construyeron {len(pasadas_df):,} pasadas.")

    print("\n[2/4] Guardando detalle de puntos instantáneos...")
    all_matched_points_con_pasada_id.to_excel(PUNTOS_INSTANTANEOS_XLSX, index=False, engine='openpyxl')
    print(f"  -> Excel puntos guardado: {PUNTOS_INSTANTANEOS_XLSX}")
    all_matched_points_con_pasada_id.reset_index(drop=True).to_feather(PUNTOS_INSTANTANEOS_FEATHER)
    print(f"  -> Feather puntos guardado: {PUNTOS_INSTANTANEOS_FEATHER}")

    print("\n[3/4] Construyendo subtramos...")
    subtramos_df = None
    if "Altitude (m)" in all_matched_points_con_pasada_id.columns:
        analizar_distribucion_subtramos(all_matched_points_con_pasada_id, MAPAS_PASADAS_DIR)
        subtramos_df = construir_subtramos(all_matched_points_con_pasada_id)

    reporte_final_pasadas, reporte_final_subtramos, tramos_geometria = enriquecer_y_guardar_reportes(pasadas_df, subtramos_df)

    print("\n[4/4] Generando visualizaciones...")
    if GENERAR_MAPA_CONSOLIDADO:
        mapa_consolidado_path = os.path.join(OUT_DIR, "mapa_consolidado_pasadas.png")
        plot_todas_las_pasadas_map(reporte_final_pasadas, all_matched_points_con_pasada_id, tramos_geometria, mapa_consolidado_path)
        print(f"  -> Mapa consolidado guardado en: {mapa_consolidado_path}")

    generar_graficos_analisis(reporte_final_pasadas, GRAFICOS_ANALISIS_DIR)
    generar_graficos_eficiencia(reporte_final_pasadas, GRAFICOS_ANALISIS_DIR)
    generar_grafico_variabilidad_distancia(reporte_final_pasadas, GRAFICOS_ANALISIS_DIR)
    generar_graficos_subtramos(reporte_final_subtramos, GRAFICOS_ANALISIS_DIR)
    calcular_y_guardar_cv_subtramos(reporte_final_subtramos, OUT_DIR)

    print("\n Proceso completado. Revisa la carpeta 'outputs'.")

if __name__ == "__main__":
    run_pasadas_analysis()