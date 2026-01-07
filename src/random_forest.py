import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

df = pd.read_csv("data/processed/training.csv")

# On importe seulement un petit échantillon pour l'instant
# df = df.sample(
#     n=1_000,
#     random_state=42
# )

labels_col = "speed"

X = df.drop([labels_col], axis=1)
y = df[labels_col]



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)



rf.fit(X_train, y_train)

# test

y_pred = rf.predict(X_test)

# performance
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