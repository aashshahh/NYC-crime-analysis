# NYC Crime Pattern Modeling
### Graduate-Level End-to-End Machine Learning & KDDM Pipeline  
**Dataset:** NYPD Arrests (2006–2024), ~6M Records

---

## 1. Project Overview
This project implements a complete Knowledge Discovery and Data Mining (KDDM) workflow using the NYPD Historic Arrests dataset (2006–2024). The objective is to transform raw arrest records into interpretable crime patterns, predictive models, and insights suitable for graduate-level academic work as well as practical decision support.

The pipeline includes:
- Data ingestion, cleaning, and ETL  
- Advanced temporal, spatial, demographic, and behavioral feature engineering  
- Exploratory crime-pattern analysis  
- Supervised felony classification  
- Unsupervised clustering and similarity modeling  
- Dimensionality reduction (PCA, t-SNE)  
- Pattern mining (Apriori, GRI)  
- Model comparison and evaluation  

---

## 2. Repository Structure


NYC-crime-analysis/
├── data/                # Raw datasets (not committed)
│   └── README.md        # How to download data
├── processed/           # Cleaned/engineered datasets (ignored)
├── notebooks/
│   ├── 1_datacleaning.ipynb     # Cleaning + ETL
│   ├── 2_eda.ipynb              # Exploratory analysis + visuals
│   └── 3_modeling.ipynb         # ML models (supervised + unsupervised)
├── src/
│   ├── allmodels.py             # Model pipelines + evaluation
│   └── utils/                   # Helper functions (optional expansion)
├── visuals/                     # Exported plots and PNGs
├── requirements.txt
├── LICENSE
└── README.md

Large datasets are intentionally excluded due to GitHub’s 100 MB file limit.

---

## 3. Dataset Description

**Source:** NYC Open Data — “NYPD Arrests (Historic)”  
**Records:** ~6,000,000  
**Time Span:** 2006–2024  

Key fields:
- Arrest date and time  
- Offense type (PD code, KY code, description)  
- Precinct, borough, jurisdiction  
- Age group, sex, race  
- Latitude, longitude  
- Offense severity (felony, misdemeanor, violation)  

Dataset download steps are provided inside `data/README.md`.

---

## 4. Data Cleaning and ETL

Real-world issues in raw data:
- Missing demographic and coordinate fields  
- Duplicate arrests  
- Non-standard text categories  
- Timestamp inconsistencies  
- Structural break after 2020  
- Severe class imbalance  

Major processing steps:
- Timestamp normalization and extraction (year, month, day, hour, weekday, weekend)  
- Deduplication and coordinate validation  
- Offense grouping and normalization  
- Demographic cleaning and bucket creation  
- Rolling windows: daily, 1-day, 7-day arrests  
- Spatial feature scaling  
- Creation of “basket strings” for similarity analysis  

The engineered dataset is stored locally but excluded from GitHub.

---

## 5. Feature Engineering

### Temporal Features
- Year, Month, Day, Hour  
- Weekday / Weekend  
- Week-of-year  
- Pre/Post 2020 flag  

### Spatial Features
- Encoded precinct  
- Encoded borough  
- Scaled latitude & longitude  

### Demographic Features
- Age buckets  
- Gender indicators  
- Cleaned race categories  

### Behavioral Features
- Daily arrest count  
- Arrests yesterday  
- Arrests last 7 days  

### Offense Features
- Grouped offense category  
- Felony flag  
- Violent offense flag  
- Basket representation for similarity measures  

---

## 6. Exploratory Data Analysis (EDA)

Key findings:
- Strong weekly and seasonal cycles  
- Sharp reduction in arrests post-2020  
- Borough differences in felony distribution  
- Evening/night peaks in many offenses  
- Rolling arrest windows correlate with felony likelihood  
- Spatial patterns show high-severity zones  

All visuals are inside `2.EDA.ipynb`.

---

## 7. Modeling

### Supervised Learning
- Logistic Regression  
- Ridge Regression  
- Naïve Bayes  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (with scaling)  
- Linear SVM  
- MLP Neural Network  

### Unsupervised & Pattern Mining
- K-Means Clustering  
- Hierarchical Clustering (dendrogram)  
- PCA (2D projection)  
- t-SNE (nonlinear embedding)  
- Apriori Association Rules  
- Generalized Rule Induction (GRI)  
- Cosine & Jaccard similarity matrices  

All modeling steps are implemented in `3.model.ipynb` and `src/allmodels.py`.

---

## 8. Model Performance Summary

| Model             | Accuracy | F1 (Felony=1) | Strength                                   | Limitation                         |
|------------------|----------|---------------|---------------------------------------------|-------------------------------------|
| Random Forest     | 0.8306   | 0.7134        | Captures non-linear patterns; strongest     | More computationally expensive      |
| Logistic Regression | 0.7493 | 0.6250        | Highly interpretable                        | Limited by linear boundary          |
| Linear SVM        | 0.7512   | 0.6257        | Good high-dimensional separation            | Still linear                        |
| MLP Classifier    | ~0.78    | ~0.67         | Learns nonlinear relationships              | Needs tuning                        |
| Decision Tree     | ~0.78    | ~0.60         | Clear rule-based model                      | High variance                       |
| Naïve Bayes       | ~0.74    | ~0.52         | Very fast baseline                          | Independence assumption             |
| KNN               | 0.7490   | ~0.43         | Intuitive neighborhood-based approach       | High-dimensional sparsity           |

---

## 9. Key Findings
- Felony likelihood is strongly shaped by offense type and recent arrest windows.  
- Crime patterns follow hourly, daily, and seasonal cycles.  
- Spatial features substantially enhance classification accuracy.  
- Behavioral features (7-day window) are among the strongest predictors.  
- Non-linear models outperform linear baselines.  
- Clustering and similarity structures confirm coherent feature separation.  

---

## 10. Limitations
- Arrest data is influenced by enforcement patterns, not total crime.  
- Class imbalance affects felony prediction performance.  
- One-hot encoded dimensions affect SVM/KNN.  
- Post-2020 drift alters temporal patterns.  
- No socioeconomic or environmental context variables included.  

---

## 11. Future Work
- Integration of weather, mobility, and socioeconomic data  
- Spatio-temporal sequence modeling (LSTM, Transformers, GNNs)  
- Drift detection and adaptive retraining pipelines  
- HDBSCAN for improved clustering  
- Crime forecasting (SARIMA, Prophet, XGBoost)  
- Interactive dashboard or API deployment  

---

## 12. Contributors
- **Aash Jatin Shah**  
- **Viraj Tapkir**  
- **Shail Shah**
