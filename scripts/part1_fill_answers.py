from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset_prueba.csv"
RESP = ROOT / "respuestas.txt"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

def find_col(df, candidates):
    # Devuelve la columna real buscando por nombre exacto o parcial (case-insensitive).
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    raise KeyError(f"No se encontró ninguna de {candidates}. Columnas: {list(df.columns)[:25]}...")

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"No se encontró {DATA}. Coloca dataset_prueba.csv en data/")
    df = pd.read_csv(DATA)

    # Normaliza el texto de la fecha
    if 'last_date_of_month' in df.columns:
        df['last_date_of_month'] = df['last_date_of_month'].astype(str).str.strip()

    # TRAIN = junio y julio; TEST = agosto (según enunciado)
    train = df[df['last_date_of_month'].isin(['6/30/2014','7/31/2014'])].copy()
    test  = df[df['last_date_of_month'].isin(['8/31/2014'])].copy()

    # Ubica columnas clave de forma robusta
    arpu_col  = find_col(train, ['ARPU','arpu'])
    rech_col  = find_col(train, ['total_rech_num','rech'])
    churn_col = find_col(train, ['churn'])
    date_col  = find_col(train, ['last_date_of_month','date_of_month'])

    # PREGUNTA 1
    # Clasificación por cuantiles 80/90 (equivale a deciles 9 y 10):
    q80 = train[arpu_col].quantile(0.8)
    q90 = train[arpu_col].quantile(0.9)
    def clasif(v):
        if pd.isna(v):           return 'normal'        # decisión segura si hay NaN
        if v >= q90:             return 'platino'
        if v >= q80:             return 'gold'
        return 'normal'
    train['clasificacion_clientes_revenue'] = train[arpu_col].apply(clasif)
    c_q1 = train['clasificacion_clientes_revenue'].value_counts().reindex(['platino','gold','normal'], fill_value=0)

    # PREGUNTA 2
    # Flag de recarga desde total_rech_num (>0 => 1)
    train['flag_recarga'] = (pd.to_numeric(train[rech_col], errors='coerce').fillna(0) > 0).astype(int)
    c_q2 = train['flag_recarga'].value_counts().reindex([1,0], fill_value=0)

    # PREGUNTA 3
    # Proporción de churn por mes (junio y julio)
    train[churn_col] = pd.to_numeric(train[churn_col], errors='coerce').fillna(0).astype(int)
    prop = train.groupby(date_col)[churn_col].mean()
    p_jun = float(prop.get('6/30/2014', np.nan))
    p_jul = float(prop.get('7/31/2014', np.nan))

    # PREGUNTA 4
    # Columnas con >70% de nulos (en TRAIN)
    null_ratio = train.isna().mean()
    cols_to_drop = null_ratio[null_ratio > 0.7].index.tolist()
    n_borradas = len(cols_to_drop)

    # PREGUNTA 5
    # Gráficos de distribución de total_rech_num por churn
    plt.figure()
    sns.boxplot(data=train, x=churn_col, y=rech_col)
    plt.title('total_rech_num por churn')
    plt.savefig(OUT / "q5_total_rech_num_boxplot.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.histplot(data=train, x=rech_col, hue=churn_col, bins=50, stat='density', common_norm=False)
    plt.title('Distribución de total_rech_num por churn')
    plt.savefig(OUT / "q5_total_rech_num_hist.png", bbox_inches='tight')
    plt.close()

    # RESPUESTAS.TXT
    content = f"""
1 {int(c_q1['platino'])} {int(c_q1['gold'])} {int(c_q1['normal'])}
2 {int(c_q2[1])} {int(c_q2[0])}
3 {p_jun:.6f} {p_jul:.6f}
4 {n_borradas}
"""
    RESP.write_text(content, encoding="utf-8")
    print("[LISTO] Parte 1 completada.")
    print(f" - respuestas.txt -> {RESP}")
    print(f" - gráficos -> {OUT/'q5_total_rech_num_boxplot.png'}, {OUT/'q5_total_rech_num_hist.png'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
