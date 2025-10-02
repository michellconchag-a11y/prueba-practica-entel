# Parte 2 – Modelado de churn

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# XGBOOST
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# VARIABLES GLOBALES
SEED = 42
ROOT = Path(__file__).resolve().parents[1]   # carpeta raíz del repo
DATA = ROOT / "data" / "dataset_prueba.csv"  # dataset
OUT  = ROOT / "outputs"                      # carpeta de resultados
OUT.mkdir(exist_ok=True, parents=True)
TARGET = "churn"                             # variable objetivo

# Leer dataset
def read_data():
    if not DATA.exists():
        raise FileNotFoundError(f"No se encontró {DATA}.")
    df = pd.read_csv(DATA)
    # Asegurar que la columna de fechas esté en formato string limpio
    df['last_date_of_month'] = df['last_date_of_month'].astype(str).str.strip()
    return df

# Split temporal: junio=train, julio=valid, agosto=test
def time_split(df):
    tr = df[df['last_date_of_month']=='6/30/2014'].copy()
    va = df[df['last_date_of_month']=='7/31/2014'].copy()
    te = df[df['last_date_of_month']=='8/31/2014'].copy()
    return tr, va, te

# Eliminar columnas con más de 70% de nulos
def drop_high_nulls(df, thr=0.7):
    null_ratio = df.isna().mean()                 # proporción de nulos por columna
    to_drop = null_ratio[null_ratio > thr].index.tolist()
    return df.drop(columns=to_drop), to_drop

# Separar features en numéricas y categóricas
def split_features(df):
    drop_cols = [c for c in ['mobile_number','last_date_of_month',TARGET] if c in df.columns]
    feats = [c for c in df.columns if c not in drop_cols]
    cat = [c for c in feats if df[c].dtype=='object']
    num = [c for c in feats if c not in cat]
    return feats, cat, num

# Preprocesamiento: imputar nulos + OneHot a categóricas
def build_pre(cat_cols, num_cols):
    return ColumnTransformer([
        # Para numéricas: imputar con mediana
        ('num', SimpleImputer(strategy='median'), num_cols),
        # Para categóricas: imputar con moda y aplicar OneHot
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# Construcción de modelos
def make_models(pre, pos_weight=None):
    models = {}
    # Modelo lineal base
    models['LogisticRegression'] = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))
    ])
    # Modelo de árboles bagging
    models['RandomForest'] = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight='balanced_subsample', random_state=SEED
        ))
    ])
    # Modelo boosting
    if HAS_XGB:
        models['XGBoost'] = Pipeline([
            ('pre', pre),
            ('clf', XGBClassifier(
                n_estimators=600,         # número de árboles
                max_depth=6,              # profundidad máxima
                learning_rate=0.05,       # tasa de aprendizaje
                subsample=0.8,            # proporción de muestras por árbol
                colsample_bytree=0.8,     # proporción de features por árbol
                reg_lambda=1.0,           # regularización L2
                random_state=SEED,
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                scale_pos_weight=pos_weight if pos_weight is not None else 1.0
            ))
        ])
    return models

# Evaluación de un modelo: métricas + curvas + threshold óptimo
def eval_model(model, X, y, name, prefix):
    proba = model.predict_proba(X)[:,1]  # probabilidades de churn
    auc = roc_auc_score(y, proba)        # ROC AUC
    ap  = average_precision_score(y, proba)  # PR AUC

    # Guardar curva ROC
    fig1, ax1 = plt.subplots()
    RocCurveDisplay.from_predictions(y, proba, ax=ax1)
    ax1.set_title(f"ROC – {name}")
    fig1.savefig(OUT / f"{prefix}_roc.png", bbox_inches='tight'); plt.close(fig1)

    # Guardar curva PR
    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax2)
    ax2.set_title(f"PR – {name}")
    fig2.savefig(OUT / f"{prefix}_pr.png", bbox_inches='tight'); plt.close(fig2)

    # Calcular umbral óptimo por F1
    prec, rec, thr = precision_recall_curve(y, proba)
    f1_vals = (2*prec*rec)/(prec+rec+1e-9)
    idx = int(np.nanargmax(f1_vals))
    best_thr = 0.5 if idx>=len(thr) else float(thr[idx])
    pred = (proba >= best_thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1]).tolist()

    return {
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "threshold": best_thr,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [0,1],
    }, proba, pred

# MAIN
def main():
    print("[1/9] Leyendo datos...")
    df = read_data()

    print("[2/9] Split temporal...")
    tr, va, te = time_split(df)
    # Asegurar que churn sea numérico
    for part in (tr, va, te):
        part[TARGET] = pd.to_numeric(part[TARGET], errors='coerce').fillna(0).astype(int)

    print("[3/9] Drop de nulos >70% usando train (junio)...")
    tr2, dropped = drop_high_nulls(tr, 0.7)
    va2 = va.drop(columns=[c for c in dropped if c in va.columns], errors='ignore')
    te2 = te.drop(columns=[c for c in dropped if c in te.columns], errors='ignore')

    print("[4/9] Seleccionando features...")
    feats, cat_cols, num_cols = split_features(tr2)
    X_tr, y_tr = tr2[feats], tr2[TARGET]
    X_va, y_va = va2[feats], va2[TARGET]
    X_te, y_te = te2[feats], te2[TARGET]

    # Calcular scale_pos_weight para XGBoost
    pos = y_tr.sum()
    neg = max(len(y_tr) - pos, 1)
    spw = float(neg / max(pos, 1))

    print("[5/9] Construyendo modelos (incluye XGBoost={})...".format(HAS_XGB))
    pre = build_pre(cat_cols, num_cols)
    models = make_models(pre, pos_weight=spw)

    print("[6/9] Entrenando...")
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)

    print("[7/9] Validación y selección por PR AUC...")
    metrics_valid = {}
    for name, mdl in models.items():
        m, _, _ = eval_model(mdl, X_va, y_va, name, f"valid_{name.lower()}")
        metrics_valid[name] = m

    with open(OUT / "metrics_valid.json", "w") as f:
        json.dump(metrics_valid, f, indent=2)

    # Elegir el mejor modelo según PR AUC
    best_name = max(metrics_valid.keys(), key=lambda k: metrics_valid[k]["pr_auc"])
    best_model = models[best_name]
    best_thr = metrics_valid[best_name]["threshold"]
    joblib.dump(best_model, OUT / f"best_model_{best_name}.joblib")

    print(f"[8/9] Evaluación final en test con {best_name} (umbral {best_thr:.4f})...")
    proba_te = best_model.predict_proba(X_te)[:,1]
    pred_te  = (proba_te >= best_thr).astype(int)

    metrics_test = {
        "model": best_name,
        "threshold": float(best_thr),
        "roc_auc": float(roc_auc_score(y_te, proba_te)),
        "pr_auc": float(average_precision_score(y_te, proba_te)),
        "f1": float(f1_score(y_te, pred_te)),
        "precision": float(precision_score(y_te, pred_te, zero_division=0)),
        "recall": float(recall_score(y_te, pred_te)),
        "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
        "n_features": int(len(feats)),
        "dropped_columns": dropped
    }
    with open(OUT / "metrics_test.json", "w") as f:
        json.dump(metrics_test, f, indent=2)

    # Guardar predicciones con ID y probabilidad
    pred_df = te2.copy()
    id_cols = ['mobile_number'] if 'mobile_number' in pred_df.columns else []
    pred_df['proba_churn'] = proba_te
    pred_df['pred_churn']  = pred_te
    pred_df = pred_df[id_cols + ['proba_churn','pred_churn', TARGET] + [c for c in pred_df.columns if c not in id_cols + [TARGET, 'proba_churn','pred_churn']]]
    pred_df.to_csv(OUT / "predicciones_test.csv", index=False)

    print("[9/9] Listo. Revisa outputs/. Modelos evaluados:", list(models.keys()))

if __name__ == "__main__":
    main()
