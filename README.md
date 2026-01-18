# 🚦 Urban Traffic Speed Prediction  
### Machine Learning applied to New York and Marseille Road Networks

## 📌 Project Overview

This project was developed as part of a university challenge.  
The goal was to build a **machine learning model capable of predicting urban road traffic speed in Marseille**, using **training data from New York City**.

The main challenge was **domain transfer**: training a model on New York data and applying it to a completely different city using only OpenStreetMap (OSM) data and feature engineering.

---

## 🗽 Data Sources (New York)

The available tools for research on New York included:

- **OpenStreetMap (OSM)**  
  Access to key geospatial data of a city

- **travel_times dataset (2013)**  
  Contains data such as the average speed per hour in 2013, with over 103 million documented observations 

- **nodes dataset**  
  - Contains data on street intersections

- **links dataset**  
  - Contains data about _links_, which are parts of a street that connects nodes

---

## ⚙️ Data Processing Pipeline

Using techniques such as **feature selection, data cleaning, data encoding, and feature engineering**, a complete pipeline was designed that is capable of:

---

### 1️⃣ Dataset Preprocessing

- Removal of unused training features

- Removal of impassable streets

- Optimization of identifiers for efficient joins  

---

### 2️⃣ Data Enrichment

- Conversion of datasets to **GeoDataFrame** format

- Association of nodes with `landuse` and `place` features
 
- Creation of new features such as `max_speed`, `lanes` and `dist_to_center` (distance to city center)

---

### 3️⃣ Dataset Postprocessing

#### 🔹 Nodes
- Z-score normalization of `dist_to_center`

- Categorization of location features

- Addition of the number of links per node

#### 🔹 Links
- Calculation of street angle

#### 🔹 Ways (grouped links of the same street)
- Conversion of `max_speed` to **km/h**

#### 🔹 Travel Times
- Extraction of features such as time of day (`hour`) and weekend indicator (`is_weekend`)

#### 🔹 Global Processing
- Static definition of maximum speed for missing values based on street type

- Grouping of streets with identicals `link_id`, `hour` and `is_weekend`, based on the average speed of each group

- Saving dataset to **CSV format**

---

## 🤖 Machine Learning Model

Using methods such as **grid search** and ** cross-validation**, a **Random Forest Regressor** was trained using **80% of the dataset**.

### ✅ Best Hyperparameters Found

- n_estimators = 200  

- min_samples_leaf = 5  

- max_features = 0.5  

- max_depth = 30  

- bootstrap = False

## 📊 Model Evaluation Metrics
The following metrics were obtained after training:

- Mean Absolute Error (MAE): 7.61 km/h

- Root Mean Squared Error (RMSE): 12.74 km/h

- Coefficient of Determination (R²): 0.472

### 🚗 Speed Statistics
- Average speed: 32.82 km/h

- Median speed: 30.96 km/h

- Minimum speed: 4.99 km/h

- Maximum speed: 104.74 km/h

These results show that the model performs particularly well around central tendency measures (mean and median), which are the most relevant for urban traffic analysis.

## 🗺️ Testing on Marseille
The trained model was then applied to Marseille, which required reconstructing the New York feature space using data extracted from Marseille OpenStreetMap (OSM).

Through feature engineering, the following attributes were recreated:

- Street centrality

- Maximum speed

- Street length

- Distance to city center

- Node connectivity

- Categorical variables (via one-hot encoding)

### 🧪 Test Configuration
- `is_weekend` = 0

- `hour` = 12

With these parameters, the model successfully generated speed predictions for Marseille streets, and the results were saved for visualization.

## 🗺️ Interactive Visualization
A Python-based solution was developed to generate an interactive HTML map displaying:
- Street names

- Predicted traffic speed per street segment

- A color legend for speed differentiation


To improve visual clarity, the lowest and highest 5% quantiles were removed. This prevents extreme values from distorting the color scale and ensures better differentiation among intermediate speeds

## 🧰 Libraries Used
- `osmnx` – Geospatial data extraction (New York & Marseille)
- `pandas` – Data import and processing
- `numpy` – Mathematical operations
- `GeoPandas` – Geospatial data manipulation
- `scikit-learn` – Model training, evaluation, and metrics
- `joblib` – Model persistence
- `folium` – Interactive HTML map generation
- `branca` – Color scale and legend creation

## 🚀 Future Improvements
Potential next steps for this project include:

- Development of a backend integrated with a modern frontend using React, Next.js, Node.js...

- Implementation of more advanced models, such as XGBoost, Graph Neural Networks (GNN) and others

- Interactive filtering by time of day and day of the week

- Exploration of new feature engineering strategies

- Testing the model with data from other years

## 🔗 Final Notes
This project highlights how machine learning combined with geospatial data can be used to model and transfer urban traffic behavior across different cities.

📌 I invite you to explore the full repository and dive deeper into the implementation.
