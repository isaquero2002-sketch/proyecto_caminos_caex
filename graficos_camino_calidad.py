# graficos_geo.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Rutas ===
BASE    = r"C:\Users\icquerov\OneDrive - Anglo American\Desktop\Proyecto_Caminos"
OUT_DIR = os.path.join(BASE, "outputs", "resultados_calidad_camino")
FIGS    = os.path.join(OUT_DIR, "figs")
os.makedirs(FIGS, exist_ok=True)

# === Carga CSVs ya generados por tu script principal ===
geo_csv      = os.path.join(OUT_DIR, "geo_consolidado.csv")
zona_mes_csv = os.path.join(OUT_DIR, "zonas_por_mes.csv")
icz_csv      = os.path.join(OUT_DIR, "Zonas_ICZ.csv")
top3_csv     = os.path.join(OUT_DIR, "zonas_top3_por_mes.csv")
tramos_csv   = os.path.join(OUT_DIR, "tramos_top_zonas_top3_por_mes.csv")

top10_consolidado_files = glob.glob(os.path.join(OUT_DIR, "top10_tramos_*_consolidado.csv"))

for p in [geo_csv, zona_mes_csv, icz_csv]:
    if not os.path.exists(p):
        raise SystemExit(f"[ERROR] Falta {p}. Corre primero el script de cálculo.")

geo      = pd.read_csv(geo_csv)
zona_mes = pd.read_csv(zona_mes_csv)
zona_w   = pd.read_csv(icz_csv)
top3     = pd.read_csv(top3_csv) if os.path.exists(top3_csv) else pd.DataFrame(columns=['MES', 'ZONA', 'pct_exig'])
tr_top   = pd.read_csv(tramos_csv) if os.path.exists(tramos_csv) else pd.DataFrame()

if top10_consolidado_files:
    tr_top_consolidado = pd.read_csv(top10_consolidado_files[0])
else:
    tr_top_consolidado = pd.DataFrame(columns=['TRAMO', 'IDC'])

# Normalizaciones básicas
for c in ["ZONA","MES","TRAMO","nivel"]:
    if c in geo.columns:
        geo[c] = geo[c].astype(str)
if "ICZ" in zona_w.columns:
    zona_w["ICZ"] = pd.to_numeric(zona_w["ICZ"], errors="coerce")

# === 1) Barras: ICZ promedio ponderado por zona (ordenado por ICZ) ===
zona_w = zona_w.sort_values(["ICZ","n_tramos"], ascending=[False, False])
plt.figure(figsize=(10,5))
plt.bar(zona_w["ZONA"], zona_w["ICZ"], color="tab:red", alpha=0.85)
plt.xticks(rotation=45, ha="right")
plt.ylabel("ICZ promedio ponderado (%)")
plt.title("Índice de Criticidad por Zona (ponderado anual)")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
f1 = os.path.join(FIGS, "B1_ICZ_promedio_zonas.png")
plt.savefig(f1, dpi=150); plt.close()
print("[OK]", f1)

# === 2) Heatmap: % exigente por mes y zona (filas ordenadas por ICZ) ===
# Creamos pivote y orden de filas = orden ICZ
orden_z = zona_w["ZONA"].tolist()
piv = zona_mes.pivot_table(index="ZONA", columns="MES", values="pct_exig", aggfunc="mean")
# ordenar filas por ICZ y columnas por el orden natural de meses (si existe)
piv = piv.reindex(index=orden_z)

try:
    import seaborn as sns
    plt.figure(figsize=(1.2*len(piv.columns)+4, 0.35*len(piv.index)+3))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label":"% exigente"})
    plt.title("% de tramos exigentes por mes y zona (orden por ICZ anual)")
    plt.tight_layout()
    f2 = os.path.join(FIGS, "B2_heatmap_pct_exig_mes_zona.png")
    plt.savefig(f2, dpi=150); plt.close()
    print("[OK]", f2)
except Exception as e:
    # fallback simple sin seaborn
    plt.figure(figsize=(10,5))
    plt.imshow(piv.values, aspect="auto", cmap="Reds")
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
    plt.yticks(range(len(piv.index)), piv.index)
    plt.colorbar(label="% exigente")
    plt.title("% de tramos exigentes por mes y zona (orden por ICZ anual)")
    plt.tight_layout()
    f2 = os.path.join(FIGS, "B2_heatmap_pct_exig_mes_zona.png")
    plt.savefig(f2, dpi=150); plt.close()
    print("[OK] (fallback matplotlib)", f2)

# === 3) Barras apiladas: composición de niveles por zona (promedio anual) ===
niv_comp = (geo
    .assign(n=1)
    .groupby(["ZONA","nivel"], as_index=False)["n"].sum())
tot_z = niv_comp.groupby("ZONA")["n"].transform("sum")
niv_comp["pct"] = niv_comp["n"]/tot_z*100
# pivot para apilar
niv_piv = niv_comp.pivot_table(index="ZONA", columns="nivel", values="pct", aggfunc="sum").fillna(0)
# ordenar filas por ICZ
niv_piv = niv_piv.reindex(index=orden_z)

niv_piv = niv_piv[["favorable","moderado","exigente"]] if all(k in niv_piv.columns for k in ["favorable","moderado","exigente"]) else niv_piv

plt.figure(figsize=(10,6))
bottom = np.zeros(len(niv_piv))
for col, color in zip(niv_piv.columns, ["tab:green","tab:orange","tab:red"]):
    plt.bar(niv_piv.index, niv_piv[col].values, bottom=bottom, label=col.capitalize(), color=color, alpha=0.85)
    bottom += niv_piv[col].values
plt.xticks(rotation=45, ha="right")
plt.ylabel("% de tramos")
plt.title("Composición de niveles por zona (promedio anual)")
plt.legend()
plt.tight_layout()
f3 = os.path.join(FIGS, "B3_composicion_niveles_por_zona.png")
plt.savefig(f3, dpi=150); plt.close()
print("[OK]", f3)

# === 4) Para la zona más crítica por ICZ: Top-10 tramos por mes (IDC) ===
if orden_z:
    top_zona = orden_z[0]
    subdir = os.path.join(FIGS, f"tramos_{top_zona}"); os.makedirs(subdir, exist_ok=True)
    meses_orden = zona_mes["MES"].unique().tolist()
    for mes in meses_orden:
        g = (geo[(geo["MES"]==mes) & (geo["ZONA"]==top_zona)]
               .drop_duplicates("TRAMO")
               .sort_values("IDC", ascending=False)
               .head(10))
        if g.empty:
            continue
        plt.figure(figsize=(12,6))
        plt.bar(g["TRAMO"].astype(str), g["IDC"].astype(float), color="tab:purple")
        plt.xticks(rotation=90)
        plt.ylabel("IDC (0–6, mayor=peor)")
        plt.title(f"{top_zona}: Top-10 tramos por IDC • {mes}")
        plt.tight_layout()
        fp = os.path.join(subdir, f"top10_tramos_{mes}.png")
        plt.savefig(fp, dpi=150); plt.close()
        print("[OK]", fp)

# === 5) (Opcional) Ranking Top-3 zonas por mes (barras agrupadas) ===
if not top3.empty:
    plt.figure(figsize=(12,6))
    # ordenar zonas por ICZ de mayor a menor para consistencia visual
    top3["ZONA"] = pd.Categorical(top3["ZONA"], categories=orden_z, ordered=True)
    top3 = top3.sort_values(["MES","ZONA"])
    for mes, g in top3.groupby("MES"):
        plt.bar(g["ZONA"].astype(str) + f" ({mes})", g["pct_exig"], label=mes)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("% exigente (Top-3 por mes)")
    plt.title("Top-3 zonas por mes (% de tramos exigentes)")
    plt.tight_layout()
    f4 = os.path.join(FIGS, "B4_top3_zonas_por_mes.png")
    plt.savefig(f4, dpi=150); plt.close()
    print("[OK]", f4)

# === 6) Gráfico Top-10 Tramos de la Zona más Crítica (Consolidado Anual) ===
if not tr_top_consolidado.empty:
    # Extraer el nombre de la zona del nombre del archivo para el título
    try:
        # Asume el formato "top10_tramos_ZONA_consolidado.csv"
        zona_critica = os.path.basename(top10_consolidado_files[0]).split('_')[2]
    except (IndexError, NameError):
        zona_critica = "Zona Más Crítica"

    # Ordenar por IDC para el gráfico
    tr_top_consolidado = tr_top_consolidado.sort_values("IDC", ascending=False)

    plt.figure(figsize=(12, 7))
    plt.bar(tr_top_consolidado["TRAMO"].astype(str), tr_top_consolidado["IDC"], color="darkslateblue", alpha=0.9)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("IDC Promedio Anual (0-6, mayor=peor)")
    plt.xlabel("Tramo")
    plt.title(f"Top 10 Tramos Más Críticos (Promedio Anual) en Zona '{zona_critica}'")
    plt.grid(True, axis="y", linestyle='--', alpha=0.6)
    plt.tight_layout()
    f6 = os.path.join(FIGS, f"B6_top10_tramos_{zona_critica}_consolidado.png")
    plt.savefig(f6, dpi=150); plt.close()
    print("[OK]", f6)

print("\nListo. Figuras en:", FIGS)
