# Imports
import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load data
kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "dawerty/cleaned-daytrading-training-data", path="data/", unzip=True
)
df = pd.read_csv("data/stock_trading.csv")

# Separate features and target
X = df.drop("is_profit", axis=1)
y = df["is_profit"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=18)
selector.fit_transform(X_train.drop(["sym", "datetime"], axis=1), y_train)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.drop(["sym", "datetime"], axis=1))
X_test = scaler.transform(X_test.drop(["sym", "datetime"], axis=1))

# KNN model
knn = KNeighborsClassifier(n_neighbors=7)

# Fit model
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate accuracy
acc = knn.score(X_test, y_test)
print(acc)
