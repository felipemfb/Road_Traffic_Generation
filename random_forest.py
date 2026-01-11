"""
Random Forest pour la prédiction de vitesse du trafic
Avec sauvegarde du modèle et des métadonnées pour la visualisation
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Création du dossier pour les modèles
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/training.csv")

labels_col = "speed"

X = df.drop([labels_col], axis=1)
y = df[labels_col]

# Sauvegarder les noms des colonnes pour la prédiction sur de nouvelles villes
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=200, 
    min_samples_leaf=5,
    max_features=0.5,
    max_depth=30,
    bootstrap=False
)

# Entraînement
print("Entraînement du modèle...")
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("------- test terminé ------------")
print("mae: ", mae)
print("rmse: ", rmse)
print("r2: ", r2)

# Statistiques sur données prédites
print("vitesse moyenne:", y_pred.mean())
print("min: ", y_pred.min())
print("max: ", y_pred.max())
print("mediane: ", np.median(y_pred))

# ============================================
# SAUVEGARDE DU MODÈLE ET DES MÉTADONNÉES
# ============================================

print("\nSauvegarde du modèle...")

# Extraire les valeurs uniques des colonnes catégorielles avant one-hot encoding
# Ces infos sont nécessaires pour reproduire le même encodage sur une nouvelle ville

# Récupérer les catégories depuis les noms de colonnes one-hot encoded
road_type_categories = [col.replace("road_type_", "") for col in feature_columns if col.startswith("road_type_")]
landuse_categories = [col.replace("landuse_", "") for col in feature_columns if col.startswith("landuse_") and not col.startswith("landuse_way_dest_")]
landuse_way_dest_categories = [col.replace("landuse_way_dest_", "") for col in feature_columns if col.startswith("landuse_way_dest_")]

# Métadonnées du modèle
model_metadata = {
    "feature_columns": feature_columns,
    "road_type_categories": road_type_categories,
    "landuse_categories": landuse_categories,
    "landuse_way_dest_categories": landuse_way_dest_categories,
    "metrics": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    },
    "speed_stats": {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max())
    }
}

# Sauvegarder le modèle
joblib.dump(rf, "models/traffic_rf_model.joblib")
print("Modèle sauvegardé: models/traffic_rf_model.joblib")

# Sauvegarder les métadonnées
joblib.dump(model_metadata, "models/model_metadata.joblib")
print("Métadonnées sauvegardées: models/model_metadata.joblib")

print("\n✓ Sauvegarde terminée!")
print(f"  - Colonnes de features: {len(feature_columns)}")
print(f"  - Types de routes: {len(road_type_categories)}")
print(f"  - Catégories landuse: {len(landuse_categories)}")
print(f"  - Catégories landuse_way_dest: {len(landuse_way_dest_categories)}")
