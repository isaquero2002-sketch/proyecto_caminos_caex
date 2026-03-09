import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
from itertools import product

# =========================
# 0) RUTAS (AJUSTA ESTO)
# =========================
BASE_PATH = Path(r"C:\Users\icquerov\OneDrive - Anglo American\Desktop\Proyecto_Caminos")

# Cambia aquí el archivo que quieras analizar:
# TR42 (con daño):
FILE_TRAMO8 = BASE_PATH / "outputs" / "analisis_regresion" / "TRamo42_mod_reg.xlsx"

# TR08 (modelo):
#FILE_TRAMO8 = BASE_PATH / "outputs" / "analisis_regresion" / "tramo08_modelo2.xlsx"

OUT_DIR = BASE_PATH / "outputs" / "analisis_regresion_resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1) HELPERS
# =========================
def to_float_comma(x):
    """Convierte ' -33,15 ' a -33.15 y soporta números ya numéricos."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
    return pd.to_numeric(x, errors="coerce")

def normalize_truck(s):
    """Normaliza nombres de camión: CDH-76 -> CDH76 para mejorar el match."""
    if pd.isna(s): return ""
    return str(s).strip().upper().replace(" ", "").replace("-", "")

def robust_group_summary(df, y_col):
    """Estadísticos robustos (mejor que promedio cuando hay outliers)."""
    return pd.Series({
        "n": df[y_col].notna().sum(),
        "promedio": df[y_col].mean(),
        "mediana": df[y_col].median(),
        "p25": df[y_col].quantile(0.25),
        "p75": df[y_col].quantile(0.75),
        "iqr": df[y_col].quantile(0.75) - df[y_col].quantile(0.25),
        "std": df[y_col].std(),
        "min": df[y_col].min(),
        "max": df[y_col].max(),
    })

def ensure_damage_binary(df, col="daño"):
    """
    Convierte daño a 0/1:
    - Si viene numérico (ej µdaño/km): >0 => 1
    - Si viene vacío para no-damage: NaN => 0
    - Si viene texto: sí/true/1 => 1
    """
    if col not in df.columns:
        df[col] = 0
        return df
    
    # Lógica: Si la celda NO está vacía (no es NaN ni string vacío) => 1 (Evento), sino 0
    df[col] = df[col].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1)
    return df

# --- NUEVOS HELPERS PARA RSM ---
def get_range(df, col, p_low=0.05, p_high=0.95):
    """Rango operativo robusto (P5–P95)."""
    return float(df[col].quantile(p_low)), float(df[col].quantile(p_high))

def make_distance_km(df):
    """Distancia km: usa dist_3d_m si existe, si no usa (fecha_fin-fecha_inicio)*Vel."""
    out = df.copy()
    if "dist_3d_m" in out.columns:
        out["dist_3d_m"] = out["dist_3d_m"].apply(to_float_comma)
        out["distancia_km"] = out["dist_3d_m"] / 1000.0
        src = "dist_3d_m"
    else:
        for c in ["fecha_inicio", "fecha_fin"]:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], errors="coerce")
        if "fecha_inicio" in out.columns and "fecha_fin" in out.columns and "Velocidad (Km/h)" in out.columns:
            out["duracion_h"] = (out["fecha_fin"] - out["fecha_inicio"]).dt.total_seconds() / 3600.0
            out["distancia_km"] = out["duracion_h"] * out["Velocidad (Km/h)"]
            src = "tiempo_x_velocidad"
        else:
            out["distancia_km"] = np.nan
            src = "sin_distancia"
    out.loc[out["distancia_km"] <= 0, "distancia_km"] = np.nan
    return out, src

def build_features_for_prediction(df_base, features, V_col="Velocidad (Km/h)", Fc_col="F. de Carga",
                                  pend_col="pendiente_local_pct", V_new=None, Fc_new=None, pend_new=None):
    """Crea un X con las mismas columnas del modelo, con posibilidad de inyectar V/Fc/pendiente."""
    tmp = df_base.copy()

    if V_new is not None:
        tmp[V_col] = V_new
    if Fc_new is not None:
        tmp[Fc_col] = Fc_new
    if pend_new is not None and pend_col in tmp.columns:
        tmp[pend_col] = pend_new

    # Recalcular interacciones SI están en el modelo
    if "velocidad_x_carga" in features and V_col in tmp.columns and Fc_col in tmp.columns:
        tmp["velocidad_x_carga"] = tmp[V_col] * tmp[Fc_col]

    if "velocidad_x_pendiente" in features and V_col in tmp.columns and pend_col in tmp.columns:
        tmp["velocidad_x_pendiente"] = tmp[V_col] * tmp[pend_col]

    Xp = tmp[features].copy()
    Xp_const = sm.add_constant(Xp, has_constant="add")
    return Xp_const

def rsm_and_savings_adaptive(df_model, model, features, out_dir, tramo_name,
                             precio_litro=530.0, grid_n=35,
                             x_var=None, y_var=None,
                             fixed=None,
                             p_levels=None,
                             clip_to_domain=True):
    """
    RSM = SUPERFICIE DE PREDICCIÓN del modelo (no DOE).
    - x_var, y_var: ejes de la superficie
    - fixed: dict con variables fijas (ej {"Pedal": 95} o {"pendiente_local_pct": 5})
    - p_levels: si quieres varias superficies por niveles de pendiente (lista de valores)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df_model.copy()

    # ---------- helpers ----------
    def get_range(df, col, p_low=0.05, p_high=0.95):
        return float(df[col].quantile(p_low)), float(df[col].quantile(p_high))

    def make_distance_km(df):
        out = df.copy()
        if "dist_3d_m" in out.columns:
            out["dist_3d_m"] = out["dist_3d_m"].apply(to_float_comma)
            out["distancia_km"] = out["dist_3d_m"] / 1000.0
        else:
            for c in ["fecha_inicio", "fecha_fin"]:
                if c in out.columns:
                    out[c] = pd.to_datetime(out[c], errors="coerce")
            if "fecha_inicio" in out.columns and "fecha_fin" in out.columns and "Velocidad (Km/h)" in out.columns:
                out["duracion_h"] = (out["fecha_fin"] - out["fecha_inicio"]).dt.total_seconds() / 3600.0
                out["distancia_km"] = out["duracion_h"] * out["Velocidad (Km/h)"]
            else:
                out["distancia_km"] = np.nan
        out.loc[out["distancia_km"] <= 0, "distancia_km"] = np.nan
        return out

    def recompute_derived(tmp):
        # Recalcula términos derivados si están en el modelo
        if "Pedal" in tmp.columns and "Pedal_sq" in features:
            tmp["Pedal_sq"] = tmp["Pedal"] ** 2
        if "pendiente_local_pct" in tmp.columns and "pendiente_local_pct_sq" in features:
            tmp["pendiente_local_pct_sq"] = tmp["pendiente_local_pct"] ** 2
        if "pendiente_local_pct" in tmp.columns and "Velocidad (Km/h)" in tmp.columns and "pend_vel" in features:
            tmp["pend_vel"] = tmp["pendiente_local_pct"] * tmp["Velocidad (Km/h)"]
        if "Pedal" in tmp.columns and "pendiente_local_pct" in tmp.columns and "pedal_x_pendiente" in features:
            tmp["pedal_x_pendiente"] = tmp["Pedal"] * tmp["pendiente_local_pct"]
        if "Pedal" in tmp.columns and "carga_ton" in tmp.columns and "pedal_x_carga_ton" in features:
            tmp["pedal_x_carga_ton"] = tmp["Pedal"] * tmp["carga_ton"]
        return tmp

    def predict_one(base_row, x, y, fixed_local):
        tmp = base_row.copy()

        tmp[x_var] = x
        tmp[y_var] = y

        for k, v in (fixed_local or {}).items():
            tmp[k] = v

        tmp = recompute_derived(tmp)

        Xp = tmp[features].copy()
        Xp_const = sm.add_constant(Xp, has_constant="add")
        return float(model.predict(Xp_const).iloc[0])

    # ---------- auto-elección de ejes por tramo ----------
    # Si no defines x_var/y_var, elige según features
    if x_var is None or y_var is None:
        # TR42 típico: features = [pendiente_local_pct, Pedal_sq, pend_vel]
        if ("pend_vel" in features) and ("pendiente_local_pct" in features) and ("Velocidad (Km/h)" in df.columns):
            x_var = "Velocidad (Km/h)"
            y_var = "pendiente_local_pct"
        # TR08 típico: features incluyen Pedal y Velocidad (Km/h)
        elif ("Pedal" in features) and ("Velocidad (Km/h)" in df.columns):
            x_var = "Velocidad (Km/h)"
            y_var = "Pedal"
        else:
            raise ValueError(f"No pude inferir ejes RSM. Define x_var/y_var manualmente. features={features}")

    fixed = fixed or {}

    # ---------- base_row: usa mediana de variables necesarias ----------
    base_row = df.iloc[[0]].copy()

    # Asegurar que las columnas base existan
    for c in set([x_var, y_var] + list(fixed.keys())):
        if c not in df.columns:
            # si falta, créala con NaN
            df[c] = np.nan
            base_row[c] = np.nan

    # si te falta Pedal pero tienes Pedal_sq en el modelo: fija Pedal por mediana si existe
    if ("Pedal_sq" in features) and ("Pedal" in df.columns) and ("Pedal" not in fixed):
        fixed["Pedal"] = float(df["Pedal"].median())

    # Pendiente por niveles (opcional)
    if p_levels is None and ("pendiente_local_pct" in df.columns) and (y_var != "pendiente_local_pct") and ("pendiente_local_pct" in features):
        # si NO es eje, lo puedes fijar por niveles
        p_levels = [
            float(df["pendiente_local_pct"].quantile(0.25)),
            float(df["pendiente_local_pct"].quantile(0.50)),
            float(df["pendiente_local_pct"].quantile(0.75)),
        ]

    levels = p_levels if p_levels is not None else [None]

    # ---------- dominio ----------
    x_min, x_max = get_range(df, x_var)
    y_min, y_max = get_range(df, y_var)
    X_grid = np.linspace(x_min, x_max, grid_n)
    Y_grid = np.linspace(y_min, y_max, grid_n)

    df = make_distance_km(df)

    # Pred base (para ahorro)
    tmp_base = df.copy()
    tmp_base = recompute_derived(tmp_base)
    Xb = sm.add_constant(tmp_base[features], has_constant="add")
    df["y_pred"] = model.predict(Xb)

    # ---------- generar superficies ----------
    for lvl in levels:
        fixed_local = dict(fixed)

        # si estamos iterando pendiente fija
        if lvl is not None:
            fixed_local["pendiente_local_pct"] = float(lvl)

        rows = []
        for x, y in product(X_grid, Y_grid):
            yhat = predict_one(base_row.copy(), x, y, fixed_local)
            rows.append({"x": x, "y": y, "yhat": yhat, "pend_fix": fixed_local.get("pendiente_local_pct", np.nan)})

        grid_df = pd.DataFrame(rows)

        # Pivot correcto para contour
        pivot = grid_df.pivot(index="y", columns="x", values="yhat")
        Xc = pivot.columns.values
        Yc = pivot.index.values
        Z = pivot.values

        # Contour
        plt.figure(figsize=(10, 7))
        cs = plt.contourf(Xc, Yc, Z, levels=25)
        plt.colorbar(cs, label="Consumo predicho (L/km)")
        plt.xlabel(x_var)
        plt.ylabel(y_var)

        title = f"Superficie de predicción (OLS) | {tramo_name} | {y_var} vs {x_var}"
        if lvl is not None and "pendiente_local_pct" in fixed_local and (x_var != "pendiente_local_pct") and (y_var != "pendiente_local_pct"):
            title += f" | Pendiente fija={fixed_local['pendiente_local_pct']:.2f}%"
        if "Pedal" in fixed_local and (x_var != "Pedal") and (y_var != "Pedal"):
            title += f" | Pedal fijo={fixed_local['Pedal']:.1f}%"
        plt.title(title)
        plt.tight_layout()

        suf = ""
        if lvl is not None:
            suf = f"pendFix{lvl:.2f}".replace(".", "p")
        if ("Pedal" in fixed_local) and (x_var != "Pedal") and (y_var != "Pedal"):
            suf += f"pedFix{fixed_local['Pedal']:.1f}".replace(".", "p")

        plt.savefig(out_dir / f"RSM_contour_{tramo_name}{y_var}_vs{x_var}{suf}.png", dpi=200)
        plt.close()

        # Guardar grid
        grid_df.to_excel(out_dir / f"RSM_grid_{tramo_name}{y_var}_vs{x_var}{suf}.xlsx", index=False)

    print(f" RSM guardada en: {out_dir}")

# =========================
# 2) CARGA EXCEL (TRAMO)
# =========================
if not FILE_TRAMO8.exists():
    raise FileNotFoundError(f"No encuentro el archivo en: {FILE_TRAMO8}")

df = pd.read_excel(FILE_TRAMO8)
df.columns = df.columns.astype(str).str.strip()

# --- FIX: Normalizar nombres de columnas (Soporte para _prom) ---
rename_map = {
    "Velocidad (Km/h)_prom": "Velocidad (Km/h)",
    "F. de Carga_prom": "F. de Carga",
    "RPM_prom": "RPM",
    "pendiente_real_pct": "pendiente_local_pct",
    "Pendiente": "pendiente_local_pct",
    "Pendiente (%)": "pendiente_local_pct",
    "daño  µdaño/km)": "daño",
    "consumo_l_km_prom": "consumo_l_km",
    "consumo_l_kmton_prom": "consumo_l_kmton",
}
df = df.rename(columns=rename_map)
# ----------------------------------------------------------------

print(" Archivo cargado:", df.shape)
print("Columnas:", list(df.columns))

# Convertir numéricos clave (soporta coma decimal)
for col in ["consumo_l_km", "pendiente_local_pct", "Velocidad (Km/h)", "Barometric Pressure (PSI)"]:
    if col in df.columns:
        df[col] = df[col].apply(to_float_comma)

# Normalizar camión si existe
if "camion" in df.columns:
    df["camion"] = df["camion"].apply(normalize_truck)

# Asegurar daño binario si existe (o crear en 0)
df = ensure_damage_binary(df, col="daño")

# Variables requeridas para el modelo
REQUIRED_VARS = ["consumo_l_km", "pendiente_local_pct", "Velocidad (Km/h)", "Barometric Pressure (PSI)"]
missing_cols_warn = [c for c in REQUIRED_VARS if c not in df.columns]
if missing_cols_warn:
    print(f"  ADVERTENCIA: faltan columnas clave: {missing_cols_warn}")

# Limpieza: Eliminar filas que no tengan datos en las variables del modelo
cols_presentes = [c for c in REQUIRED_VARS if c in df.columns]
df = df.dropna(subset=cols_presentes).copy()

print(" Data limpia:", df.shape)

# =========================
# 2.1) BLOQUE DAMAGE (RESUMEN + TEST + POTENCIAL)
# =========================
print("\n --- Análisis Damage vs No-Damage ---")

target = "consumo_l_km"
if "daño" not in df.columns:
    df["daño"] = 0

if target not in df.columns:
    print(f" Error Crítico: La columna '{target}' no se encuentra en el archivo.")
    print(f"   Columnas disponibles: {list(df.columns)}")
    raise KeyError(f"Falta columna target: {target}")

df_dmg = df[df["daño"] == 1].copy()
df_nodmg = df[df["daño"] == 0].copy()

sum_dmg = robust_group_summary(df_dmg, target)
sum_nodmg = robust_group_summary(df_nodmg, target)

# % sobreconsumo usando MEDIANA (robusto)
sobreconsumo_pct = np.nan
if pd.notna(sum_nodmg["mediana"]) and sum_nodmg["mediana"] != 0 and pd.notna(sum_dmg["mediana"]):
    sobreconsumo_pct = (sum_dmg["mediana"] - sum_nodmg["mediana"]) / sum_nodmg["mediana"] * 100

# Mann–Whitney (si hay suficiente data)
u_stat, p_value = np.nan, np.nan
if len(df_dmg) >= 5 and len(df_nodmg) >= 5:
    u_stat, p_value = stats.mannwhitneyu(
        df_dmg[target].dropna(),
        df_nodmg[target].dropna(),
        alternative="two-sided"
    )

print(f"  -> Sin daño: n={int(sum_nodmg['n'])} | mediana={sum_nodmg['mediana']:.3f} L/km")
print(f"  -> Con daño: n={int(sum_dmg['n'])} | mediana={sum_dmg['mediana']:.3f} L/km")
print(f"  -> Sobreconsumo (mediana): {sobreconsumo_pct:.2f}%")
print(f"  -> Mann–Whitney p-value: {p_value}")

# Guardar resumen damage
damage_resumen = pd.DataFrame({
    "grupo": ["SIN_DAÑO", "CON_DAÑO"],
    "n": [sum_nodmg["n"], sum_dmg["n"]],
    "promedio_L_km": [sum_nodmg["promedio"], sum_dmg["promedio"]],
    "mediana_L_km": [sum_nodmg["mediana"], sum_dmg["mediana"]],
    "p25": [sum_nodmg["p25"], sum_dmg["p25"]],
    "p75": [sum_nodmg["p75"], sum_dmg["p75"]],
    "iqr": [sum_nodmg["iqr"], sum_dmg["iqr"]],
})

damage_indicadores = pd.DataFrame({
    "metrica": ["sobreconsumo_pct_mediana", "mannwhitney_u", "mannwhitney_pvalue"],
    "valor": [sobreconsumo_pct, u_stat, p_value]
})

damage_out = OUT_DIR / f"damage_resumen_{FILE_TRAMO8.stem}.xlsx"
with pd.ExcelWriter(damage_out, engine="openpyxl") as writer:
    damage_resumen.to_excel(writer, sheet_name="resumen", index=False)
    damage_indicadores.to_excel(writer, sheet_name="indicadores", index=False)

print(f" Resumen damage guardado en: {damage_out.name}")

# =========================
# 3) ANÁLISIS DE REGRESIÓN
# =========================
print("\n --- Iniciando Análisis de Regresión ---")

df_model = df.copy()
print(f"  -> Usando dataset completo ({len(df_model)} registros).")

# --- CREACIÓN DE INTERACCIONES ---
if "Pedal" in df_model.columns and "pendiente_local_pct" in df_model.columns:
    df_model["pedal_x_pendiente"] = df_model["Pedal"] * df_model["pendiente_local_pct"]

if "Pedal " in df_model.columns and "Barometric Pressure (PSI)" in df_model.columns:
    df_model["pedal_x_barometric"] = df_model["Pedal"] * df_model["Barometric Pressure (PSI)"]

if "pendiente_local_pct" in df_model.columns and "Velocidad (Km/h)" in df_model.columns:
    df_model["pend_vel"] = df_model["pendiente_local_pct"] * df_model["Velocidad (Km/h)"]   

if "pendiente_local_pct" in df_model.columns and "pendiente_local_pct" in df_model.columns:
    df_model["pendiente_local_pct_sq"] = df_model["pendiente_local_pct"] ** 2

if "pendiente_local_pct" in df_model.columns and "Barometric Pressure (PSI)" in df_model.columns:
    df_model["pendiente_local_pct_x_barometric"] = df_model["pendiente_local_pct"] * df_model["Barometric Pressure (PSI)"]

if "Pedal" in df_model.columns and "carga_ton" in df_model.columns:
    df_model["pedal_x_carga_ton"] = df_model["Pedal"] * df_model["carga_ton"]

if "pendiente_local_pct" in df_model.columns and "carga_ton" in df_model.columns:
    df_model["pendiente_local_pct_x_carga_ton"] = df_model["pendiente_local_pct"] * df_model["carga_ton"]

# --- CREACIÓN DE INTERACCIONES / TÉRMINOS ---
if "Pedal" in df_model.columns:
    df_model["Pedal_sq"] = df_model["Pedal"] ** 2

if "pendiente_local_pct" in df_model.columns:
    df_model["pendiente_local_pct_sq"] = df_model["pendiente_local_pct"] ** 2

if "pendiente_local_pct" in df_model.columns and "Velocidad (Km/h)" in df_model.columns:
    df_model["pend_vel"] = df_model["pendiente_local_pct"] * df_model["Velocidad (Km/h)"]

if "Pedal" in df_model.columns and "pendiente_local_pct" in df_model.columns:
    df_model["pedal_x_pendiente"] = df_model["Pedal"] * df_model["pendiente_local_pct"]


# --- CONFIGURACIÓN DE VARIABLES ---
# Detectar tramo del nombre del archivo para elegir variables
filename_lower = FILE_TRAMO8.name.lower()

if "tr08" in filename_lower or "tramo08" in filename_lower:
    print(" MODO DETECTADO: T08 (Modelo cuadrático completo)")

    FEATURES = [
        # --------- LINEALES ----------
        #"pendiente_local_pct",
        "Pedal",
        #"Barometric Pressure (PSI)",
        #"carga_ton",
        "Velocidad (Km/h)",
        "pendiente_local_pct_sq",
        #"pedal_x_pendiente",
        # "Pedal_sq"
        #"Pedal_x_carga_ton",
        #"F. de Carga",
         #"pend_vel",
         #"pendiente_local_pct_x_barometric",


    ]

elif "tr42" in filename_lower or "tramo42" in filename_lower:
    print(" MODO DETECTADO: T42 (Modelo cuadrático completo)")

    FEATURES = [
     # --------- LINEALES ----------
        "pendiente_local_pct",
        #"Pedal",
        #"Barometric Pressure (PSI)",
         #"carga_ton",
        #"Velocidad (Km/h)",
        #"pendiente_local_pct_sq",
        #"pedal_x_pendiente",
        #"Pedal_sq",
        "pend_vel",
       #"Pedal_x_carga_ton",
       #"pendiente_local_pct_x_Barometric Pressure (PSI)",
       "F. de Carga",
       #"daño (µdaño/km)",

    ]

else:
    print(" No se detectó tramo específico (TR08/TR42). Usando configuración TR08.")

    FEATURES = [
        "pendiente_local_pct",
        "Pedal",
        "pendiente_local_pct_x_Pedal",
        "pendiente_local_pct_sq",
        "Pedal_sq",
    ]
target = "consumo_l_km"

available_features = [c for c in FEATURES if c in df_model.columns]

if target in df_model.columns and len(available_features) > 0:
    df_model = df_model.dropna(subset=available_features + [target])

    if len(df_model) > 10:
        X = df_model[available_features]
        y = df_model[target]

        # 2. Matriz de Correlación (Exploratorio)
        corr_matrix = df_model[available_features + [target]].corr()
        print("\nMatriz de Correlación:")
        print(corr_matrix[target].sort_values(ascending=False))
        corr_matrix.to_excel(OUT_DIR / f"matriz_correlacion_{FILE_TRAMO8.stem}.xlsx")

        # 3A) Modelo OLS BASE
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        print("\n" + "="*60)
        print("RESULTADOS DEL MODELO BASE (sin daño)")
        print("="*60)
        print(model.summary())

        with open(OUT_DIR / f"resumen_modelo_base_{FILE_TRAMO8.stem}.txt", "w") as f:
            f.write(model.summary().as_text())

        # 3B) Modelo OLS EXTENDIDO (+ daño) si hay variación en daño
        model_dmg = None
        if "daño" in df_model.columns and df_model["daño"].nunique() > 1:
            X2 = df_model[available_features + ["daño"]].copy()
            X2_const = sm.add_constant(X2)
            model_dmg = sm.OLS(y, X2_const).fit()

            print("\n" + "="*60)
            print("RESULTADOS DEL MODELO + DAÑO")
            print("="*60)
            print(model_dmg.summary())

            with open(OUT_DIR / f"resumen_modelo_con_dano_{FILE_TRAMO8.stem}.txt", "w") as f:
                f.write(model_dmg.summary().as_text())

        # 4) Verificación de Supuestos (usando el modelo base para diagnósticos)
        print("\n --- Verificación de Supuestos ---")

        # A) Multicolinealidad (VIF)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        print("\nFactor de Inflación de Varianza (VIF):")
        print(vif_data)
        vif_data.to_excel(OUT_DIR / f"analisis_vif_{FILE_TRAMO8.stem}.xlsx", index=False)

        # B) Normalidad de Residuos (Jarque-Bera)
        residuals = model.resid
        jb_stat, jb_p = stats.jarque_bera(residuals)
        print(f"\nTest Jarque-Bera: Estadístico={jb_stat:.2f}, p-value={jb_p:.4f}")
        if jb_p < 0.05:
            print("  -> Residuos NO normales (p < 0.05).")

        # C) Gráficos de Diagnóstico
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title("Histograma de Residuos")
        plt.savefig(OUT_DIR / f"diagnostico_1_histograma_{FILE_TRAMO8.stem}.png")
        plt.close()

        fig_qq = sm.qqplot(residuals, line='45', fit=True)
        plt.title("Q-Q Plot (Normalidad)")
        plt.savefig(OUT_DIR / f"diagnostico_2_qqplot_{FILE_TRAMO8.stem}.png")
        plt.close(fig_qq)

        plt.figure(figsize=(10, 6))
        plt.scatter(model.fittedvalues, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residuos vs Predichos")
        plt.xlabel("Predichos"); plt.ylabel("Residuos")
        plt.savefig(OUT_DIR / f"diagnostico_3_residuos_vs_predichos_{FILE_TRAMO8.stem}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(residuals.values, alpha=0.5)
        plt.title("Residuos vs Orden")
        plt.savefig(OUT_DIR / f"diagnostico_4_residuos_vs_orden_{FILE_TRAMO8.stem}.png")
        plt.close()

        # 5) Rango de Validez
        print("\n --- Rango de Validez del Modelo ---")
        validity_df = X.describe(percentiles=[0.05, 0.5, 0.95]).T[['min', '5%', 'mean', '95%', 'max']]
        validity_df.columns = ['Min', 'P5', 'Promedio', 'P95', 'Max']
        print(validity_df)
        validity_df.to_excel(OUT_DIR / f"rango_validez_modelo_{FILE_TRAMO8.stem}.xlsx")

        # 6) Sensibilidad y Regla 80/20
        print("\n --- Análisis de Sensibilidad (Importancia Relativa 80/20) ---")

        params = model.params.drop('const', errors='ignore')
        X_std = X.std()
        y_std = y.std()

        common_vars = params.index.intersection(X_std.index)

        sensitivity_df = pd.DataFrame({
            'Coeficiente_Regresion': params[common_vars],
            'Desv_Std_Variable': X_std[common_vars],
            'Impacto_Unitario_Std': params[common_vars] * X_std[common_vars]
        })

        sensitivity_df['Beta_Estandarizado'] = sensitivity_df['Impacto_Unitario_Std'] / y_std
        sensitivity_df['Importancia_Abs'] = sensitivity_df['Beta_Estandarizado'].abs()
        total_importance = sensitivity_df['Importancia_Abs'].sum()
        sensitivity_df['Importancia_Relativa_%'] = (sensitivity_df['Importancia_Abs'] / total_importance) * 100
        sensitivity_df = sensitivity_df.sort_values('Importancia_Relativa_%', ascending=False)
        sensitivity_df['Importancia_Acumulada_%'] = sensitivity_df['Importancia_Relativa_%'].cumsum()

        print(sensitivity_df[['Beta_Estandarizado', 'Importancia_Relativa_%', 'Importancia_Acumulada_%']])
        sensitivity_df.to_excel(OUT_DIR / f"analisis_sensibilidad_80_20_{FILE_TRAMO8.stem}.xlsx")

        plt.figure(figsize=(12, 7))
        ax1 = sns.barplot(x=sensitivity_df.index, y=sensitivity_df['Importancia_Relativa_%'])
        ax1.set_ylabel('Importancia Relativa (%)')
        ax1.set_xticks(range(len(sensitivity_df.index)))
        ax1.set_xticklabels(sensitivity_df.index, rotation=45, ha='right')

        ax2 = ax1.twinx()
        sns.lineplot(x=sensitivity_df.index, y=sensitivity_df['Importancia_Acumulada_%'], ax=ax2, marker='o', sort=False)
        ax2.set_ylabel('Importancia Acumulada (%)')
        ax2.set_ylim(0, 110)
        ax2.axhline(80, color='grey', linestyle='--', linewidth=1)
        plt.title(f"Pareto Sensibilidad - Objetivo: {target}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"sensibilidad_pareto_80_20_{FILE_TRAMO8.stem}.png")
        plt.close()

        # 7) Detección de Anomalías (Outliers) con MAD Robusto (usando modelo base)
        print("\n --- Detección de Anomalías (Outliers) ---")

        df_model['y_pred'] = model.predict(X_const)
        df_model['residuo'] = df_model[target] - df_model['y_pred']

        mediana_resid = df_model['residuo'].median()
        mad = np.median(np.abs(df_model['residuo'] - mediana_resid))
        sigma_mad = 1.4826 * mad
        umbral = 3 * sigma_mad

        print(f"  -> Mediana Residuos: {mediana_resid:.4f}")
        print(f"  -> MAD: {mad:.4f} (Sigma est: {sigma_mad:.4f})")
        print(f"  -> Umbral corte: |residuo - mediana| > {umbral:.4f}")

        df_model['es_anomalo'] = np.abs(df_model['residuo'] - mediana_resid) > umbral
        df_model['magnitud_anomalia'] = np.abs(df_model['residuo'] - mediana_resid)

        outliers_df = df_model[df_model['es_anomalo']].sort_values('magnitud_anomalia', ascending=False)
        out_name = f"outliers_{FILE_TRAMO8.stem}.xlsx"
        outliers_df.to_excel(OUT_DIR / out_name, index=False)
        print(f" Anomalías: {len(outliers_df)} ({len(outliers_df)/len(df_model)*100:.2f}%). Guardado: {out_name}")

        


        # 8) Export extra: data con pred/resid (+ daño si existía)
        extra_cols = []
        for c in ["pasada_id", "subtramo_id", "camion", "tramo", "Direccion_Subtramo",
                  "pendiente_local_pct", "Velocidad (Km/h)", "F. de Carga", "daño",
                  "fecha_inicio", "fecha_fin", "latitud", "longitud"]:
            if c in df_model.columns:
                extra_cols.append(c)

        export_cols = extra_cols + [target] + available_features + ["y_pred", "residuo", "es_anomalo", "magnitud_anomalia"]
        export_cols = [c for c in export_cols if c in df_model.columns]

        df_export = df_model[export_cols].copy()
        data_out = OUT_DIR / f"data_con_pred_resid_{FILE_TRAMO8.stem}.xlsx"
        df_export.to_excel(data_out, index=False)
        print(f" Data con predicción/residuo guardada en: {data_out.name}")

        # 9) Mini-resumen comparativo de modelos (base vs +daño)
        if model_dmg is not None:
            comparacion = pd.DataFrame({
                "modelo": ["base_sin_dano", "extendido_con_dano"],
                "R2_adj": [model.rsquared_adj, model_dmg.rsquared_adj],
                "AIC": [model.aic, model_dmg.aic],
                "BIC": [model.bic, model_dmg.bic],
                "beta_dano": [np.nan, model_dmg.params.get("daño", np.nan)],
                "pvalue_dano": [np.nan, model_dmg.pvalues.get("daño", np.nan)],
            })
            comparacion.to_excel(OUT_DIR / f"comparacion_modelos_dano_{FILE_TRAMO8.stem}.xlsx", index=False)
            print(f" Comparación modelos (+daño) guardada.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import statsmodels.api as sm

# ============================================================
# HELPERS (CLAVE PARA NO TENER SHAPES NOT ALIGNED)
# ============================================================
def exog_from_dict_like_model(model, data_dict):
    """
    Construye un DataFrame exog con EXACTAMENTE las columnas del modelo (incluye const si aplica).
    """
    exog_names = list(model.model.exog_names)  # ej: ['const','Pedal','Velocidad (Km/h)','pendiente_local_pct_sq']
    n = len(next(iter(data_dict.values())))
    X = pd.DataFrame(index=range(n))

    for name in exog_names:
        if name == "const":
            X[name] = 1.0
        else:
            if name not in data_dict:
                raise KeyError(f"Falta columna '{name}' para predecir. El modelo exige: {exog_names}")
            X[name] = data_dict[name]

    return X

def ensure_distance_km(df):
    out = df.copy()
    if "dist_3d_m" in out.columns:
        out["dist_3d_m"] = out["dist_3d_m"].apply(to_float_comma)
        out["distancia_km"] = out["dist_3d_m"] / 1000.0
        print("  -> Usando 'dist_3d_m' para distancia.")
    else:
        if "fecha_inicio" in out.columns and "fecha_fin" in out.columns and "Velocidad (Km/h)" in out.columns:
            out["fecha_inicio"] = pd.to_datetime(out["fecha_inicio"], errors="coerce")
            out["fecha_fin"]    = pd.to_datetime(out["fecha_fin"], errors="coerce")
            out["duracion_h"] = (out["fecha_fin"] - out["fecha_inicio"]).dt.total_seconds() / 3600.0
            out["distancia_km"] = out["Velocidad (Km/h)"] * out["duracion_h"]
            print("  -> Distancia = Velocidad * Duración (no existe dist_3d_m).")
        else:
            out["distancia_km"] = np.nan
            print("  -> No pude calcular distancia (faltan columnas).")

    out.loc[out["distancia_km"] <= 0, "distancia_km"] = np.nan
    return out

def apply_steepest_descent_optimization(df, model, control_vars, step_sizes, bounds):
    """
    Aplica optimización de Máximo Descenso (Steepest Descent) para minimizar consumo.
    Dirección: -Gradiente (g = -∇f).
    """
    df_opt = df.copy()
    epsilon = 1e-4
    
    # Función interna para predecir recalculando derivadas
    def predict_wrapper(df_in):
        tmp = df_in.copy()
        # Recalcular interacciones conocidas (Sincronizar con FEATURES del modelo)
        if "Pedal" in tmp.columns:
            tmp["Pedal_sq"] = tmp["Pedal"] ** 2
        if "pendiente_local_pct" in tmp.columns:
            tmp["pendiente_local_pct_sq"] = tmp["pendiente_local_pct"] ** 2
        if "pendiente_local_pct" in tmp.columns and "Velocidad (Km/h)" in tmp.columns:
            tmp["pend_vel"] = tmp["pendiente_local_pct"] * tmp["Velocidad (Km/h)"]
        if "Pedal" in tmp.columns and "pendiente_local_pct" in tmp.columns:
            tmp["pedal_x_pendiente"] = tmp["Pedal"] * tmp["pendiente_local_pct"]
        
        # Construir exog
        exog_names = list(model.model.exog_names)
        X_matrix = pd.DataFrame(index=tmp.index)
        for name in exog_names:
            if name == "const":
                X_matrix[name] = 1.0
            elif name in tmp.columns:
                X_matrix[name] = tmp[name]
            else:
                X_matrix[name] = 0.0 # Fallback seguro
        return model.predict(X_matrix)

    # 1. Calcular Gradientes Numéricos (∇f)
    gradients = {}
    for var in control_vars:
        # f(x + eps)
        df_plus = df_opt.copy(); df_plus[var] += epsilon
        y_plus = predict_wrapper(df_plus)
        # f(x - eps)
        df_minus = df_opt.copy(); df_minus[var] -= epsilon
        y_minus = predict_wrapper(df_minus)
        # Central difference
        gradients[var] = (y_plus - y_minus) / (2 * epsilon)

    # 2. Aplicar Desplazamiento en dirección del Gradiente Negativo
    for var in control_vars:
        grad = gradients[var]
        step = step_sizes.get(var, 0.0)
        
        # Dirección: Si grad > 0, reducir variable. Si grad < 0, aumentar.
        direction = np.where(grad > 0, -1.0, 1.0)
        
        new_val = df_opt[var] + (direction * step)
        vmin, vmax = bounds.get(var, (-np.inf, np.inf))
        df_opt[f"{var}_opt"] = new_val.clip(lower=vmin, upper=vmax)
        df_opt[f"delta_{var}"] = df_opt[f"{var}_opt"] - df_opt[var]

    # 3. Predicción Final
    df_final_calc = df_opt.copy()
    for var in control_vars:
        df_final_calc[var] = df_final_calc[f"{var}_opt"]
    
    df_opt["y_pred_mej"] = predict_wrapper(df_final_calc)
    return df_opt

# ============================================================
# 8) AHORRO REAL (TR08 / TR42) - SOLO VARIABLES CONTROLABLES
# ============================================================
print("\n --- Estimación de Ahorro REAL (Litros y $) ---")
 
# Conversión de 593 USD/m3 a CLP/L:
# 593 USD/m3 = 0.593 USD/L. Asumiendo 1 USD = 980 CLP => 0.593 * 980 = 581.14 CLP/L
PRECIO_LITRO = 581  # CLP/L (ajusta)

PRECIO_LITRO = 530  # CLP/L (ajusta)

df_calc = ensure_distance_km(df_model)

# Litros reales observados
df_calc["litros_reales"] = df_calc["consumo_l_km"] * df_calc["distancia_km"]

# Predicción base del modelo (sobre datos reales)
# Construimos X_base con columnas exactas del modelo:
base_dict = {}
exog_names = list(model.model.exog_names)

# armar dict desde df_calc para todas las columnas requeridas (excepto const)
for name in exog_names:
    if name == "const":
        continue
    if name not in df_calc.columns:
        # Si el modelo usa una feature creada (ej pend_vel), la calculamos aquí
        if name == "pend_vel":
            df_calc["pend_vel"] = df_calc["pendiente_local_pct"] * df_calc["Velocidad (Km/h)"]
        elif name == "Pedal_sq":
            df_calc["Pedal_sq"] = df_calc["Pedal"] ** 2
        else:
            raise KeyError(f"Tu modelo requiere '{name}' y no existe en df_model.")

    base_dict[name] = df_calc[name].values

X_base = exog_from_dict_like_model(model, base_dict)
df_calc["y_pred"] = model.predict(X_base)

# --------- Escenario de mejora (Optimización Steepest Descent) ----------
filename_lower = FILE_TRAMO8.name.lower()
tag = ""

if "tr08" in filename_lower or "tramo08" in filename_lower:
    # TR08: Controlables = Pedal y Velocidad
    control_vars = ["Pedal", "Velocidad (Km/h)"]
    step_sizes = {"Pedal": 2.0, "Velocidad (Km/h)": 1.0} # Magnitud del paso
    bounds = {"Pedal": (0, 100), "Velocidad (Km/h)": (0, 60)}
    
    print(" Aplicando Máximo Descenso para TR08 (Pedal, Velocidad)...")
    df_opt = apply_steepest_descent_optimization(df_calc, model, control_vars, step_sizes, bounds)
    
    # Mapear resultados al dataframe principal
    df_calc["y_pred_mej"] = df_opt["y_pred_mej"]
    df_calc["Pedal_mej"] = df_opt["Pedal_opt"]
    df_calc["Velocidad_mej"] = df_opt["Velocidad (Km/h)_opt"]
    tag = "TR08_SteepestDescent"

elif "tr42" in filename_lower or "tramo42" in filename_lower:
    # TR42: Controlables = Velocidad y F. de Carga
    control_vars = ["Velocidad (Km/h)"]
    step_sizes = {"Velocidad (Km/h)": 1.0}
    bounds = {"Velocidad (Km/h)": (0, 60)}
    
    if "F. de Carga" in df_calc.columns:
        control_vars.append("F. de Carga")
        step_sizes["F. de Carga"] = 2.0
        bounds["F. de Carga"] = (0, 100)

    print(" Aplicando Máximo Descenso para TR42...")
    df_opt = apply_steepest_descent_optimization(df_calc, model, control_vars, step_sizes, bounds)
    
    df_calc["y_pred_mej"] = df_opt["y_pred_mej"]
    df_calc["Velocidad_mej"] = df_opt["Velocidad (Km/h)_opt"]
    tag = "TR42_SteepestDescent"
else:
    raise ValueError("No detecté TR08 o TR42 desde el nombre del archivo.")

# Ahorro
df_calc["ahorro_l_km"] = df_calc["y_pred"] - df_calc["y_pred_mej"]
df_calc["litros_ahorrados"] = (df_calc["ahorro_l_km"] * df_calc["distancia_km"]).clip(lower=0)
df_calc["costo_ahorrado"] = df_calc["litros_ahorrados"] * PRECIO_LITRO

# Export detalle + resumen
out_det = OUT_DIR / f"ahorro_real_detalle_{FILE_TRAMO8.stem}_{tag}.xlsx"
df_calc.to_excel(out_det, index=False)

group_cols = [c for c in ["tramo", "camion", "Direccion_Subtramo", "subtramo_id"] if c in df_calc.columns]
resumen = (df_calc.groupby(group_cols, dropna=False)
           .agg(distancia_km=("distancia_km","sum"),
                litros_reales=("litros_reales","sum"),
                litros_ahorrados=("litros_ahorrados","sum"),
                costo_ahorrado=("costo_ahorrado","sum"),
                n=("consumo_l_km","size"))
           .reset_index())

resumen["ahorro_%"] = 100 * (resumen["litros_ahorrados"] / resumen["litros_reales"])
out_res = OUT_DIR / f"ahorro_real_resumen_{FILE_TRAMO8.stem}_{tag}.xlsx"
resumen.to_excel(out_res, index=False)

print(f" Detalle ahorro real guardado en: {out_det.name}")
print(f" Resumen ahorro real guardado en: {out_res.name}")

# ============================================================
# 10) RSM TR42 - HACEMOS LO CORRECTO:
#     NO dejamos pendiente fija "porque sí".
#     La tratamos como ESCENARIO (bajo/medio/alto) o graficamos Pend vs Vel con Fc fijo (mediana).
# ============================================================
def rsm_TR42_pend_vs_vel(df_model, model, out_dir, tramo_name, grid_n=35):
    """
    RSM TR42: Pendiente vs Velocidad.
    Genera superficies para niveles de F. de Carga (Baja, Media, Alta) si existe.
    """
    required = ["pendiente_local_pct", "Velocidad (Km/h)"]
    missing = [c for c in required if c not in df_model.columns]
    if missing:
        print(f" No puedo hacer RSM TR42: faltan columnas {missing}")
        return

    exog_names = list(model.model.exog_names)

    Pmin = float(df_model["pendiente_local_pct"].quantile(0.05))
    Pmax = float(df_model["pendiente_local_pct"].quantile(0.95))
    Vmin = float(df_model["Velocidad (Km/h)"].quantile(0.05))
    Vmax = float(df_model["Velocidad (Km/h)"].quantile(0.95))

    P_grid = np.linspace(Pmin, Pmax, grid_n)
    V_grid = np.linspace(Vmin, Vmax, grid_n)

    # Niveles de F. de Carga
    fc_levels = []
    if "F. de Carga" in df_model.columns and df_model["F. de Carga"].notna().any():
        fc_levels = [
            ("Baja", float(df_model["F. de Carga"].quantile(0.25))),
            ("Media", float(df_model["F. de Carga"].quantile(0.50))),
            ("Alta", float(df_model["F. de Carga"].quantile(0.75)))
        ]
    else:
        fc_levels = [("Fija", 0.0)]

    for label_fc, Fc_fix in fc_levels:
        rows = []
        for p, v in product(P_grid, V_grid):
            data_dict = {}

            for name in exog_names:
                if name == "const":
                    continue
                
                if name == "pendiente_local_pct":
                    data_dict[name] = np.array([p])
                elif name == "Velocidad (Km/h)":
                    data_dict[name] = np.array([v])
                elif name == "pend_vel":
                    data_dict[name] = np.array([p * v])
                elif name == "F. de Carga":
                    data_dict[name] = np.array([Fc_fix])
                elif name == "Pedal_sq":
                    # Robust Pedal fix
                    if "Pedal" in df_model.columns and df_model["Pedal"].notna().any():
                        ped_fix = float(df_model["Pedal"].median())
                    else:
                        ped_fix = 95.0
                    data_dict[name] = np.array([ped_fix**2])
                else:
                    # Fallback
                    if name in df_model.columns:
                        data_dict[name] = np.array([float(df_model[name].median())])
                    else:
                        data_dict[name] = np.array([0.0])

            Xp = exog_from_dict_like_model(model, data_dict)
            yhat = float(model.predict(Xp)[0])
            rows.append({"Pendiente": p, "Velocidad": v, "Consumo_pred": max(yhat, 0)})

        grid = pd.DataFrame(rows)
        
        pivot = grid.pivot(index="Pendiente", columns="Velocidad", values="Consumo_pred")
        plt.figure(figsize=(10, 7))
        cs = plt.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=25)
        plt.colorbar(cs, label="Consumo predicho (L/km)")
        plt.xlabel("Velocidad (km/h)")
        plt.ylabel("Pendiente (%)")
        
        extra = f" | Fc ({label_fc})={Fc_fix:.1f}"
        plt.title(f"TR42 | Pendiente vs Velocidad{extra}")
        plt.tight_layout()
        plt.savefig(out_dir / f"RSM_TR42_Pend_vs_Vel_{tramo_name}_{label_fc}.png", dpi=200)
        plt.close()

        best = grid.loc[grid["Consumo_pred"].idxmin()]
        print(f" TR42 ({label_fc}) Óptimo: Pend={best['Pendiente']:.2f}% | V={best['Velocidad']:.2f} km/h | Consumo={best['Consumo_pred']:.2f}")

# Ejecutar RSM solo para TR42
def rsm_TR08_pedal_vs_vel(df_model, model, out_dir, tramo_name, grid_n=35):
    """
    RSM TR08: Pedal vs Velocidad (variables de control), fijando Pendiente en la mediana.
    """
    required = ["Pedal", "Velocidad (Km/h)"]
    missing = [c for c in required if c not in df_model.columns]
    if missing:
        print(f" No puedo hacer RSM TR08: faltan columnas {missing}")
        return

    exog_names = list(model.model.exog_names)

    # Rangos para ejes (P5-P95)
    Ped_min = float(df_model["Pedal"].quantile(0.05))
    Ped_max = float(df_model["Pedal"].quantile(0.95))
    Vmin = float(df_model["Velocidad (Km/h)"].quantile(0.05))
    Vmax = float(df_model["Velocidad (Km/h)"].quantile(0.95))

    Ped_grid = np.linspace(Ped_min, Ped_max, grid_n)
    V_grid = np.linspace(Vmin, Vmax, grid_n)

    # Definir niveles de Pendiente (Baja, Media, Alta) para generar múltiples superficies
    pend_levels = []
    if "pendiente_local_pct" in df_model.columns:
        pend_levels = [
            ("Baja", float(df_model["pendiente_local_pct"].quantile(0.25))),
            ("Media", float(df_model["pendiente_local_pct"].quantile(0.50))),
            ("Alta", float(df_model["pendiente_local_pct"].quantile(0.75)))
        ]
    else:
        pend_levels = [("Fija", 0.0)]
    
    # F. de Carga fija (si existe)
    Fc_fix = None
    if "F. de Carga" in df_model.columns:
        Fc_fix = float(df_model["F. de Carga"].median())

    # Iterar sobre cada nivel de pendiente
    for label_pend, Pend_fix in pend_levels:
        rows = []
        for ped, v in product(Ped_grid, V_grid):
            data_dict = {}
            
            for name in exog_names:
                if name == "const":
                    continue
                
                # Ejes
                if name == "Pedal":
                    data_dict[name] = np.array([ped])
                elif name == "Velocidad (Km/h)":
                    data_dict[name] = np.array([v])
                
                # Derivadas / Fijas
                elif name == "Pedal_sq":
                    data_dict[name] = np.array([ped**2])
                elif name == "pendiente_local_pct":
                    data_dict[name] = np.array([Pend_fix])
                elif name == "pendiente_local_pct_sq":
                    data_dict[name] = np.array([Pend_fix**2])
                elif name == "pend_vel":
                    data_dict[name] = np.array([Pend_fix * v])
                elif name == "F. de Carga":
                    data_dict[name] = np.array([Fc_fix])
                elif name == "pedal_x_pendiente":
                    data_dict[name] = np.array([ped * Pend_fix])
                else:
                    # Fallback: mediana
                    if name in df_model.columns:
                        data_dict[name] = np.array([float(df_model[name].median())])
                    else:
                        data_dict[name] = np.array([0.0]) 

            Xp = exog_from_dict_like_model(model, data_dict)
            yhat = float(model.predict(Xp)[0])
            rows.append({"Pedal": ped, "Velocidad": v, "Consumo_pred": max(yhat, 0)})

        grid = pd.DataFrame(rows)
        # Guardar grid individual si se desea, o solo el plot
        # grid.to_excel(out_dir / f"RSM_TR08_grid_{tramo_name}_{label_pend}.xlsx", index=False)

        pivot = grid.pivot(index="Pedal", columns="Velocidad", values="Consumo_pred")
        plt.figure(figsize=(10, 7))
        cs = plt.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=25)
        plt.colorbar(cs, label="Consumo predicho (L/km)")
        plt.xlabel("Velocidad (km/h)")
        plt.ylabel("Pedal (%)")
        
        extra = f" | Pendiente ({label_pend})={Pend_fix:.2f}%"
        plt.title(f"TR08 | Pedal vs Velocidad{extra}")
        plt.tight_layout()
        plt.savefig(out_dir / f"RSM_TR08_Pedal_vs_Vel_{tramo_name}_{label_pend}.png", dpi=200)
        plt.close()
        
        best = grid.loc[grid["Consumo_pred"].idxmin()]
        print(f" TR08 ({label_pend}) Óptimo: Pedal={best['Pedal']:.2f}% | V={best['Velocidad']:.2f} km/h | Consumo={best['Consumo_pred']:.2f}")

# Ejecutar RSM según el tramo detectado
if "tr42" in filename_lower or "tramo42" in filename_lower:
    rsm_TR42_pend_vs_vel(df_model, model, OUT_DIR, FILE_TRAMO8.stem, grid_n=35)
elif "tr08" in filename_lower or "tramo08" in filename_lower:
    rsm_TR08_pedal_vs_vel(df_model, model, OUT_DIR, FILE_TRAMO8.stem, grid_n=35)
else:
    print(" No se detectó TR08 ni TR42 en el nombre del archivo. No se generó RSM específica.")