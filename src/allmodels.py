# --------------------------------------------------------
# NYC Arrest Data — Logistic Regression, Random Forest, SVM,
# PCA + UMAP Dimensionality Reduction, Cosine & Jaccard Similarity
# Target: is_felony (0/1)
# --------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
    HAS_UMAP = True
except:
    HAS_UMAP = False

# --------------------------------------------------------
# 1. Load the data
# --------------------------------------------------------
df = pd.read_csv("/Users/shailshah/Downloads/processed/processed_arrests_main.csv")

# Select usable features
features = [
    "ARREST_BORO", "AGE_GROUP", "PERP_SEX", "PERP_RACE",
    "Latitude", "Longitude", "offense_group"
]

target = "is_felony"

df = df[features + [target]]
df.dropna(inplace=True)

# --------------------------------------------------------
# 2. Encode categorical variables
# --------------------------------------------------------
categorical_cols = ["ARREST_BORO", "AGE_GROUP", "PERP_SEX", "PERP_RACE", "offense_group"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# --------------------------------------------------------
# 3. Train-test split
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# scale only numeric columns
scaler = StandardScaler()
numeric_cols = ["Latitude", "Longitude"]
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ========================================================
# 4A. Logistic Regression
# ========================================================
print("\n================ Logistic Regression ================\n")

logreg = LogisticRegression(max_iter=1500)
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_lr))
print("\nClassification Report:\n", classification_report(y_test, pred_lr))


# ========================================================
# 4B. Random Forest
# ========================================================
print("\n================ Random Forest ================\n")

rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    min_samples_split=40,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_rf))
print("\nClassification Report:\n", classification_report(y_test, pred_rf))


# ========================================================
# 4C. SVM (light tuning)
# ========================================================
print("\n================ SVM Classifier ================\n")

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_svm))
print("\nClassification Report:\n", classification_report(y_test, pred_svm))


# ========================================================
# 5. Dimensionality Reduction
# ========================================================
print("\n================ Dimensionality Reduction ================\n")

# Use numeric + encoded features
X_dr = df_encoded.drop(columns=[target])

# scale for DR
scaler_dr = StandardScaler()
X_scaled = scaler_dr.fit_transform(X_dr)

# PCA 2D
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 6))
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], s=4, alpha=0.4)
plt.title("PCA (2D) – Arrest Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=300)
plt.close()

print("PCA Explained Variance:", pca.explained_variance_ratio_)

# UMAP
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1, random_state=42)
    umap_2d = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    plt.scatter(umap_2d[:, 0], umap_2d[:, 1], s=4, alpha=0.4)
    plt.title("UMAP (2D) – Arrest Embeddings")
    plt.tight_layout()
    plt.savefig("umap_2d.png", dpi=300)
    plt.close()
else:
    print("UMAP not installed – skipping.")


# ========================================================
# 6. Cosine + Jaccard Similarity (PD offense embeddings)
# ========================================================
print("\n================ Similarity Analysis ================\n")

# For similarity → use offense_group only
df_sim = df[["offense_group"]].dropna().copy()

# One-hot encode
mat = pd.get_dummies(df_sim["offense_group"])

# Cosine similarity across offense types
cos_sim = cosine_similarity(mat.T)
cos_df = pd.DataFrame(cos_sim, index=mat.columns, columns=mat.columns)

plt.figure(figsize=(8,7))
sns.heatmap(cos_df, cmap="viridis")
plt.title("Cosine Similarity – Offense Groups")
plt.tight_layout()
plt.savefig("cosine_offense.png", dpi=300)
plt.close()

# Jaccard similarity
bin_mat = (mat.T > 0).astype(int)
offenses = bin_mat.index
jaccard = pd.DataFrame(0.0, index=offenses, columns=offenses)

for i in offenses:
    for j in offenses:
        inter = np.logical_and(bin_mat.loc[i], bin_mat.loc[j]).sum()
        union = np.logical_or(bin_mat.loc[i], bin_mat.loc[j]).sum()
        jaccard.loc[i, j] = inter / union if union > 0 else 0

plt.figure(figsize=(8,7))
sns.heatmap(jaccard, cmap="magma")
plt.title("Jaccard Similarity – Offense Groups")
plt.tight_layout()
plt.savefig("jaccard_offense.png", dpi=300)
plt.close()


print("\nAll models + DR + similarity complete.\n")
