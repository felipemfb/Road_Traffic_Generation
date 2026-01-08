import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

df = pd.read_csv("data/processed/training.csv")

# On importe seulement un petit échantillon pour l'instant
# df = df.sample(
#     n=200_000,
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
    random_state=42,
    n_jobs=-1,
    n_estimators=200, 
    min_samples_leaf=5,
    max_features=0.5,
    max_depth=30,
    bootstrap=False
)

# # Sélection des hyperparams avec grid-search + cross validation

# param_dist = {
#     "n_estimators": [200, 300, 500],
#     "max_depth": [None, 15, 20, 30],
#     "min_samples_leaf": [1, 5, 10],
#     "max_features": ["sqrt", 0.5, 0.7],
#     "bootstrap": [True, False]
# }

# search = RandomizedSearchCV(
#     rf,
#     param_distributions=param_dist,
#     n_iter=30,
#     scoring="r2",
#     cv=3,
#     verbose=2,
#     n_jobs=-1,
#     random_state=42
# )

# search.fit(X_train, y_train)

# print("Meilleurs hyperparamètres :", search.best_params_)
# print("Meilleur score CV R²   :", search.best_score_)

# best_rf = search.best_estimator_
# y_pred = best_rf.predict(X_test)

# test

rf.fit(X_train, y_train)

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