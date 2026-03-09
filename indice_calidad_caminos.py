# -*- coding: utf-8 -*-
# SOLO GEOMETRÍA · Índice de Dificultad de Camino (IDC) por zona/mes y tramos
# Salidas: CSVs y PNGs en outputs/

import os, re, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
BASE     = r"C:\Users\icquerov\OneDrive - Anglo American\Desktop\Proyecto_Caminos"
GEO_DIR  = os.path.join(BASE, "Geometría")
OUT_DIR  = os.path.join(BASE, "outputs")
FIGS_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ------------- HELPERS -------------
def normtxt(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"\s+"," ", s).strip().lower()
    return s

def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".").strip()
    return pd.to_numeric(x, errors="coerce")

# mes desde nombre de archivo (Enero… Dic, ene… dic, 01..12)
MESES_ES = {
    "ene":"01","enero":"01",
    "feb":"02","febrero":"02",
    "mar":"03","marzo":"03",
    "abr":"04","abril":"04",
    "may":"05","mayo":"05",
    "jun":"06","junio":"06",
    "jul":"07","julio":"07",
    "ago":"08","agosto":"08",
    "sep":"09","septiembre":"09","set":"09",
    "oct":"10","octubre":"10",
    "nov":"11","noviembre":"11",
    "dic":"12","diciembre":"12",
}

def periodo_desde_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    nlow = normtxt(name)
    # 1) AAAA separador MM
    m = re.search(r"(20\d{2})[^\d]?([01]?\d)\b", nlow)
    if m:
        y, mm = m.group(1), m.group(2).zfill(2)
        if "01" <= mm <= "12":
            return f"{y}-{mm}"
    # 2) mes texto + año
    for key, mm in MESES_ES.items():
        if re.search(rf"\b{key}\b", nlow):
            m2 = re.search(r"(20\d{2})", nlow)
            if m2:
                return f"{m2.group(1)}-{mm}"
    # 3) MM + AAAA separados
    m3 = re.search(r"\b([01]?\d)\b.*\b(20\d{2})\b", nlow)
    if m3:
        mm = m3.group(1).zfill(2); yy = m3.group(2)
        if "01" <= mm <= "12":
            return f"{yy}-{mm}"
    return None

# ------------- CARGA LIMPIA -------------
# Mapeo flexible por si cambia levemente el nombre
COLMAP = {
    "tramo": ["tramo","tramo id","nombre tramo","id tramo"],
    "pend_mayor": ["pendiente mayor del tramo","pendiente mayor","pendiente mayor(%)","pendiente mayor (%)"],
    "pend_menor": ["pendiente menor del tramo","pendiente menor","pendiente menor(%)","pendiente menor (%)"],
    "radio_int":  ["radio cuvatura interno","radio curvatura interno","radio interno"],
    "radio_ext":  ["radio curvatura externo","radio cuvatura externo","radio externo"],
    "peralte_int":["peralte interno"],
    "peralte_ext":["peralte externo"],
    "ancho_m":    ["ancho(m)","ancho m","ancho","ancho (m)"],
    "recta":      ["recta"],
    "des_circular":["des.circular","des circular","descircular","curva","circular"],
    "fecha_actualizacion":["fecha actualizacion","fecha actualización","fecha de actualizacion"]
}

def pick_col(df, variants):
    norm = {normtxt(c): c for c in df.columns}
    for v in variants:
        if normtxt(v) in norm:
            return norm[normtxt(v)]
    return None

def load_geo_file(path):
    """Lee hoja única con header=0. Quita 'Unnamed', normaliza nombres y añade PERIODO/ZONA."""
    try:
        df = pd.read_excel(path, header=0)
    except PermissionError:
        print(f"[LOCK] Cierra el Excel: {os.path.basename(path)}")
        return None
    # quita Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    if df.empty:
        return None

    # renombrar
    new = {}
    for dest, opts in COLMAP.items():
        c = pick_col(df, opts)
        if c:
            new[c] = dest
    df = df.rename(columns=new).copy()

    # garantizar columnas (aunque sea NaN)
    need = ["tramo","pend_mayor","pend_menor","radio_int","radio_ext",
            "peralte_int","peralte_ext","ancho_m","recta","des_circular","fecha_actualizacion"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan

    # tipos numéricos
    for c in ["pend_mayor","pend_menor","radio_int","radio_ext","peralte_int","peralte_ext","ancho_m"]:
        df[c] = to_float(df[c])

    # booleanos
    def truthy(x):
        s = str(x).strip().lower()
        return s in {"1","true","t","si","sí","y","yes","verdadero"}
    def falsy(x):
        s = str(x).strip().lower()
        return s in {"0","false","f","no","nan","none",""}

    if df["recta"].notna().any():
        df["recta"] = df["recta"].apply(truthy)
    else:
        df["recta"] = False  # por defecto no-recta sólo si no hay info? -> mejor: False = curva conservadora

    if df["des_circular"].notna().any():
        df["des_circular"] = df["des_circular"].apply(truthy)
    else:
        df["des_circular"] = False

    # fecha
    df["fecha_actualizacion"] = pd.to_datetime(df["fecha_actualizacion"], errors="coerce")

    # tramo canon + zona
    df["TRAMO_CAN"] = df["tramo"].astype(str).str.strip().str.upper()
    # ZONA = primer token hasta espacio (INF5 TR08 -> INF5)
    df["ZONA"] = df["TRAMO_CAN"].str.extract(r"^([A-Z0-9]+)")

    # periodo
    per = periodo_desde_filename(path)
    if per is None and df["fecha_actualizacion"].notna().any():
        per = (df["fecha_actualizacion"].dt.year.astype("Int64").astype(str) + "-" +
               df["fecha_actualizacion"].dt.month.astype("Int64").astype(str).str.zfill(2))
        # si hay varias, toma la moda
        per = per.mode().iloc[0] if not per.mode().empty else None
    df["PERIODO"] = per if per else "SIN_PERIODO"

    df["__source_file"] = os.path.basename(path)
    return df

# ------------- ÍNDICE DE DIFICULTAD (IDC) -------------
def compute_scores_row(r):
    # pendiente efectiva (máximo absoluto de mayor/menor)
    p_vals = [r.get("pend_mayor"), r.get("pend_menor")]
    p_vals = [v for v in p_vals if pd.notna(v)]
    pendiente = np.nanmax(np.abs(p_vals)) if p_vals else np.nan
    # ¿es recta?
    es_recta = bool(r.get("recta")) or (not bool(r.get("des_circular")))
    # radio efectivo (mínimo) si curva; recta => 0 (no penaliza)
    if es_recta:
        radio = np.inf
    else:
        ri, rext = r.get("radio_int"), r.get("radio_ext")
        radios = [v for v in [ri, rext] if pd.notna(v)]
        radio = np.nanmin(radios) if radios else np.nan
    # peralte efectivo (máximo) si curva; recta => 0 (no penaliza)
    if es_recta:
        peralte = np.nan
    else:
        pi, pe = r.get("peralte_int"), r.get("peralte_ext")
        per_lst = [v for v in [pi, pe] if pd.notna(v)]
        peralte = np.nanmax(per_lst) if per_lst else np.nan

    # ancho
    ancho = r.get("ancho_m")

    # Scores discretos (0=mejor, 2=peor)
    # Pendiente
    if pd.isna(pendiente):
        s_p = np.nan
    elif pendiente <= 5:
        s_p = 0
    elif pendiente <= 10:
        s_p = 1
    else:
        s_p = 2

    # Curvatura (radio chico = peor). En recta => 0
    if es_recta:
        s_r = 0
    elif pd.isna(radio) or not np.isfinite(radio):
        s_r = np.nan
    elif radio > 150:
        s_r = 0
    elif radio >= 60:
        s_r = 1
    else:
        s_r = 2

    # Peralte (bajo = peor). En recta => 0
    if es_recta:
        s_e = 0
    elif pd.isna(peralte):
        s_e = np.nan
    elif peralte >= 6:
        s_e = 0
    elif peralte >= 3:
        s_e = 1
    else:
        s_e = 2

    # Ancho (angosto = peor) — opcional
    if pd.isna(ancho):
        s_w = np.nan
    elif ancho >= 10:
        s_w = 0
    elif ancho >= 8:
        s_w = 1
    else:
        s_w = 2

    # IDC con pesos dinámicos (si falta un componente, se renormaliza)
    comps = []
    weights = []
    for val, w in [(s_p, 0.35), (s_r, 0.40), (s_e, 0.20), (s_w, 0.05)]:
        if pd.notna(val):
            comps.append(val); weights.append(w)
    if comps:
        idc = float(np.average(comps, weights=weights))
    else:
        idc = np.nan

    return pd.Series({
        "pendiente_pct": pendiente,
        "radio_m": (None if np.isinf(radio) else radio),
        "peralte_pct": peralte,
        "score_pend": s_p,
        "score_radio": s_r,
        "score_peralte": s_e,
        "score_ancho": s_w,
        "IDC": idc
    })

# ------------- PIPELINE -------------
if __name__ == "__main__":
    files = [os.path.join(GEO_DIR, f)
             for f in os.listdir(GEO_DIR)
             if f.lower().endswith(".xlsx")]

    print(f"[SCAN] Excels en geometria/: {len(files)}")
    if not files:
        raise SystemExit("No hay Excel en geometria/. Verifica la ruta.")

    frames = []
    for fp in files:
        df = load_geo_file(fp)
        if df is None or df.empty:
            print(f"[WARN] Saltado: {os.path.basename(fp)}")
            continue
        frames.append(df)

    if not frames:
        raise SystemExit("No se pudo cargar ningún archivo.")

    geo = pd.concat(frames, ignore_index=True)

    # Calcula IDC y campos auxiliares
    scores = geo.apply(compute_scores_row, axis=1)
    geo2 = pd.concat([geo, scores], axis=1)

    # Guarda consolidado
    out_csv = os.path.join(OUT_DIR, "geo_idc_consolidado.csv")
    geo2.to_csv(out_csv, index=False, encoding="utf-8")
    print("[OK] Consolidado:", out_csv)

    # ----------------- AGREGADOS POR ZONA Y MES -----------------
    # Promedio de IDC por ZONA y PERIODO (mes)
    g_zm = (geo2.groupby(["PERIODO","ZONA"], dropna=False)["IDC"]
                 .mean().reset_index().rename(columns={"IDC":"IDC_prom"}))
    g_z  = (geo2.groupby(["ZONA"], dropna=False)["IDC"]
                 .mean().reset_index().rename(columns={"IDC":"IDC_prom"}))

    g_zm.to_csv(os.path.join(OUT_DIR, "idc_por_zona_mes.csv"), index=False, encoding="utf-8")
    g_z.to_csv(os.path.join(OUT_DIR, "idc_por_zona_prom_anual.csv"), index=False, encoding="utf-8")
    print("[OK] Tablas por zona guardadas.")

    # ----------------- GRÁFICOS ZONAS POR MES -----------------
    for per, sub in g_zm.groupby("PERIODO"):
        if per == "SIN_PERIODO":
            continue
        sub = sub.sort_values("IDC_prom", ascending=False)
        plt.figure(figsize=(11,5))
        plt.bar(sub["ZONA"].astype(str), sub["IDC_prom"])
        plt.title(f"IDC promedio por ZONA · {per}")
        plt.xlabel("ZONA"); plt.ylabel("IDC (0=mejor, 2=peor)")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        fpath = os.path.join(FIGS_DIR, f"ZONAS_IDC_{per}.png")
        plt.savefig(fpath, dpi=150); plt.close()
        print("[OK]", fpath)

    # Promedio anual (sobre todos los meses cargados)
    g_z = g_z.sort_values("IDC_prom", ascending=False)
    plt.figure(figsize=(11,5))
    plt.bar(g_z["ZONA"].astype(str), g_z["IDC_prom"])
    plt.title("IDC promedio por ZONA · (promedio anual)")
    plt.xlabel("ZONA"); plt.ylabel("IDC (0=mejor, 2=peor)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, "ZONAS_IDC_promedio_anual.png")
    plt.savefig(fpath, dpi=150); plt.close()
    print("[OK]", fpath)


    

    # ----------------- TRAMOS de la 2ª PEOR ZONA POR MES -----------------
    # Para cada mes, toma la segunda zona con mayor IDC_prom y grafica sus tramos
    for per, sub in g_zm.groupby("PERIODO"):
        if per == "SIN_PERIODO" or len(sub) < 2:
            continue
        sub = sub.sort_values("IDC_prom", ascending=False)
        zona_peor2 = sub.iloc[1]["ZONA"]

        # IDC por TRAMO en esa zona y mes
        ztr = (geo2.loc[(geo2["PERIODO"]==per) & (geo2["ZONA"]==zona_peor2)]
                      .groupby("TRAMO_CAN")["IDC"].mean().reset_index()
                      .sort_values("IDC", ascending=False))

        if ztr.empty:
            continue

        # Colorear por terciles (sólo para orientar la vista)
        q1, q2 = ztr["IDC"].quantile([0.33, 0.66])
        def bucket(v):
            if v <= q1: return "bajo"
            if v <= q2: return "medio"
            return "alto"
        ztr["bucket"] = ztr["IDC"].apply(bucket)

        color_map = {"alto":"tab:red","medio":"tab:orange","bajo":"tab:green"}
        colors = ztr["bucket"].map(color_map)

        plt.figure(figsize=(12,6))
        plt.bar(ztr["TRAMO_CAN"].astype(str), ztr["IDC"], color=colors)
        plt.title(f"Tramos en 2ª peor ZONA ({zona_peor2}) · {per}")
        plt.xlabel("TRAMO"); plt.ylabel("IDC (0=mejor, 2=peor)")
        plt.xticks(rotation=90); plt.tight_layout()
        fpath = os.path.join(FIGS_DIR, f"TRAMOS_segundaZona_{per}.png")
        plt.savefig(fpath, dpi=150); plt.close()
        print("[OK]", fpath)

    print("\nListo. Revisa:", FIGS_DIR)
