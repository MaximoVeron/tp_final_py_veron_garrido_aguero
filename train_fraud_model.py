"""
Script optimizado para entrenar modelo de detección de fraudes.
Genera modelos RandomForest e IsolationForest con evaluación completa.
"""

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, auc)
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("INICIANDO ENTRENAMIENTO DE MODELO DE DETECCIÓN DE FRAUDES")
print("="*60)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1/8] Cargando datos...")
df = pd.read_csv("data.csv")
print(f"  ✓ Datos cargados: {df.shape}")

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO
# ============================================================================
print("\n[2/8] Análisis exploratorio...")
fraudes_por_tipo = df.groupby('type')['isFraud'].agg(['count', 'sum'])
fraudes_por_tipo.columns = ['Total_Transacciones', 'Cantidad_Fraudes']
fraudes_por_tipo['%_Fraude'] = (fraudes_por_tipo['Cantidad_Fraudes'] / fraudes_por_tipo['Total_Transacciones']) * 100

print(f"  Total de transacciones: {len(df):,}")
print(f"  Total de fraudes: {df['isFraud'].sum():,} ({(df['isFraud'].sum()/len(df))*100:.4f}%)")
print(f"  Fraudes por tipo:")
for tipo, row in fraudes_por_tipo.iterrows():
    print(f"    - {tipo}: {int(row['Cantidad_Fraudes']):,} ({row['%_Fraude']:.2f}%)")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n[3/8] Ingeniería de features...")
df_mod = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

# Crear features derivadas
df_mod['amount_to_balance_ratio'] = df_mod['amount'] / (df_mod['oldbalanceOrg'] + 1)
df_mod['new_balance_ratio'] = (df_mod['newbalanceOrig'] - df_mod['oldbalanceOrg']) / (df_mod['amount'] + 1)
df_mod['empty_orig'] = (df_mod['oldbalanceOrg'] == 0).astype(int)
df_mod['empty_dest'] = (df_mod['oldbalanceDest'] == 0).astype(int)
df_mod['new_dest_zero'] = (df_mod['newbalanceDest'] == 0).astype(int)

df_mod = df_mod.drop(["nameOrig", "nameDest"], axis=1)
df_mod = pd.get_dummies(df_mod, columns=["type"], dtype=int)

print(f"  Features creadas: {df_mod.shape[1]}")
print(f"  Datos filtrados: {len(df_mod):,} transacciones")
print(f"  Fraudes en datos filtrados: {df_mod['isFraud'].sum():,} ({(df_mod['isFraud'].sum()/len(df_mod))*100:.4f}%)")

# ============================================================================
# 4. ANÁLISIS DE CORRELACIÓN
# ============================================================================
print("\n[4/8] Analizando correlaciones...")
fraud_corr = df_mod.corr()['isFraud'].sort_values(ascending=False)
print("  Top 5 features correlacionados:")
for feat, corr in fraud_corr[1:6].items():
    print(f"    - {feat}: {corr:.4f}")

# ============================================================================
# 5. PREPARACIÓN DE DATOS
# ============================================================================
print("\n[5/8] Preparando datos...")
x = df_mod.drop(columns=["isFraud", "isFlaggedFraud"])
y = df_mod["isFraud"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"  Entrenamiento: {x_train.shape[0]:,} ({y_train.sum():,} fraudes)")
print(f"  Prueba: {x_test.shape[0]:,} ({y_test.sum():,} fraudes)")
print(f"  Features: {x_train.shape[1]}")

# ============================================================================
# 6. ENTRENAR ISOLATION FOREST
# ============================================================================
print("\n[6/8] Entrenando Isolation Forest...")
contamination_rate = y_train.sum() / len(y_train)
print(f"  Tasa de contaminación: {contamination_rate:.6f}")

iforest = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=contamination_rate,
    n_jobs=-1,
    random_state=42
)
iforest.fit(x_train)

if_scores_test = iforest.decision_function(x_test)
if_predictions = iforest.predict(x_test)
if_preds_binary = np.where(if_predictions == -1, 1, 0)

print(f"  ✓ Isolation Forest entrenado")

# ============================================================================
# 7. ENTRENAR RANDOM FOREST
# ============================================================================
print("\n[7/8] Entrenando Random Forest...")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight=class_weight_dict,
    n_jobs=-1,
    random_state=42,
    max_features='sqrt'
)
rf_model.fit(x_train, y_train)

rf_probs = rf_model.predict_proba(x_test)[:, 1]
rf_preds = rf_model.predict(x_test)

print(f"  ✓ Random Forest entrenado")

# ============================================================================
# 8. EVALUACIÓN DE MODELOS
# ============================================================================
print("\n[8/8] Evaluando modelos...\n")

# Isolation Forest metrics
cm_if = confusion_matrix(y_test, if_preds_binary)
tn_if, fp_if, fn_if, tp_if = cm_if.ravel()

accuracy_if = accuracy_score(y_test, if_preds_binary)
precision_if = precision_score(y_test, if_preds_binary)
recall_if = recall_score(y_test, if_preds_binary)
f1_if = f1_score(y_test, if_preds_binary)
roc_auc_if = roc_auc_score(y_test, -if_scores_test)

# Random Forest metrics
cm_rf = confusion_matrix(y_test, rf_preds)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

accuracy_rf = accuracy_score(y_test, rf_preds)
precision_rf = precision_score(y_test, rf_preds)
recall_rf = recall_score(y_test, rf_preds)
f1_rf = f1_score(y_test, rf_preds)
roc_auc_rf = roc_auc_score(y_test, rf_probs)

# ========================================
# COMPARACIÓN
# ========================================
print("="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)

comparison = pd.DataFrame({
    'Modelo': ['Isolation Forest', 'Random Forest'],
    'Accuracy': [accuracy_if, accuracy_rf],
    'Precision': [precision_if, precision_rf],
    'Recall': [recall_if, recall_rf],
    'F1-Score': [f1_if, f1_rf],
    'ROC-AUC': [roc_auc_if, roc_auc_rf]
})

print("\n", comparison.to_string(index=False))

# ========================================
# ISOLATION FOREST DETAILS
# ========================================
print("\n" + "="*60)
print("ISOLATION FOREST - DETALLES")
print("="*60)
print(f"\nMatriz de Confusión:")
print(f"  Verdaderos Negativos:  {tn_if:,}")
print(f"  Falsos Positivos:      {fp_if:,}")
print(f"  Falsos Negativos:      {fn_if:,}")
print(f"  Verdaderos Positivos:  {tp_if:,}")
print(f"\nMétricas:")
print(f"  Accuracy:  {accuracy_if:.4f}")
print(f"  Precision: {precision_if:.4f}")
print(f"  Recall:    {recall_if:.4f} ← MÉTRICA CRÍTICA")
print(f"  F1-Score:  {f1_if:.4f}")
print(f"  ROC-AUC:   {roc_auc_if:.4f}")

# ========================================
# RANDOM FOREST DETAILS
# ========================================
print("\n" + "="*60)
print("RANDOM FOREST - DETALLES")
print("="*60)
print(f"\nMatriz de Confusión:")
print(f"  Verdaderos Negativos:  {tn_rf:,}")
print(f"  Falsos Positivos:      {fp_rf:,}")
print(f"  Falsos Negativos:      {fn_rf:,}")
print(f"  Verdaderos Positivos:  {tp_rf:,}")
print(f"\nMétricas:")
print(f"  Accuracy:  {accuracy_rf:.4f}")
print(f"  Precision: {precision_rf:.4f}")
print(f"  Recall:    {recall_rf:.4f} ← MÉTRICA CRÍTICA")
print(f"  F1-Score:  {f1_rf:.4f}")
print(f"  ROC-AUC:   {roc_auc_rf:.4f}")

print(f"\nReporte de Clasificación Random Forest:")
print(classification_report(y_test, rf_preds, target_names=['Legítimo', 'Fraude']))

# Feature importance
print("Top 10 Features más importantes:")
feature_importance = pd.DataFrame({
    'feature': x.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# ========================================
# SELECCIONAR MEJOR MODELO
# ========================================
print("\n" + "="*60)
print("MODELO SELECCIONADO")
print("="*60)

if recall_rf > recall_if:
    print(f"\n✓ Random Forest seleccionado")
    print(f"  - Recall {recall_rf:.4f} > Isolation Forest {recall_if:.4f}")
    best_model = rf_model
    model_name = "RandomForest"
else:
    print(f"\n✓ Isolation Forest seleccionado")
    best_model = iforest
    model_name = "IsolationForest"

# ========================================
# GUARDAR MODELOS
# ========================================
print("\n" + "="*60)
print("GUARDANDO MODELOS")
print("="*60)

os.makedirs("models", exist_ok=True)

# Guardar Random Forest
rf_path = "models/fraud_model.pkl"
with open(rf_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"✓ Random Forest guardado: {rf_path}")

# Guardar Isolation Forest
if_path = "models/isolation_forest_model.pkl"
with open(if_path, 'wb') as f:
    pickle.dump(iforest, f)
print(f"✓ Isolation Forest guardado: {if_path}")

# Guardar scaler
scaler_path = "models/scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler guardado: {scaler_path}")

# Guardar metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'n_features': x_train.shape[1],
    'feature_names': x.columns.tolist(),
    'accuracy': float(accuracy_rf),
    'recall': float(recall_rf),
    'precision': float(precision_rf),
    'f1_score': float(f1_rf),
    'roc_auc': float(roc_auc_rf),
    'contamination_rate': float(contamination_rate),
    'fraud_count_test': int(y_test.sum()),
    'frauds_detected': int(tp_rf)
}

metadata_path = "models/metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata guardado: {metadata_path}")

# ========================================
# RESUMEN FINAL
# ========================================
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"""
✓ MODELO DE DETECCIÓN DE FRAUDES OPTIMIZADO

Rendimiento Final:
  • Accuracy:  {accuracy_rf:.4f} (correctitud general)
  • Precision: {precision_rf:.4f} (no bloquear clientes legítimos)
  • Recall:    {recall_rf:.4f} (detectar fraudes reales)
  • F1-Score:  {f1_rf:.4f} (balance)
  • ROC-AUC:   {roc_auc_rf:.4f} (discriminación)

Rendimiento en Test:
  • Fraudes detectados: {tp_rf:,} de {y_test.sum():,} ({(tp_rf/y_test.sum())*100:.2f}%)
  • Falsos positivos: {fp_rf:,} ({(fp_rf/tn_rf)*100:.4f}% de transacciones legítimas)

Los modelos están listos para producción en backend/main.py
""")

print("="*60)
print("✓ ENTRENAMIENTO COMPLETADO")
print("="*60)
