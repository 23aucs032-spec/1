import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import graphviz


st.set_page_config(page_title="streamlit", layout="wide")

st.sidebar.title("Model Type")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Regression", "Classification", "Clustering"]
)

# -----------------------------------------
# SIDEBAR OPTIONS WITH MAX DEPTH
# -----------------------------------------
if model_type == "Regression":
    algorithm = st.sidebar.selectbox(
        "Regression Algorithm",
        ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor",
         "Decision Tree Regressor", "KNN Regressor"]
    )
    test_size_display = st.sidebar.slider("Test Size (%)", 10, 50)
    test_size = test_size_display / 100

    if algorithm in ["Decision Tree Regressor", "Random Forest Regressor"]:
        max_depth = st.sidebar.number_input("Max Depth", 1, 50, 5)
    else:
        max_depth = None

elif model_type == "Classification":
    algorithm = st.sidebar.selectbox(
        "Classification Algorithm",
        ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier",
         "Decision Tree Classifier", "KNN Classifier"]
    )
    test_size_display = st.sidebar.slider("Test Size (%)", 10, 50)
    test_size = test_size_display / 100

    if algorithm in ["Decision Tree Classifier", "Random Forest Classifier"]:
        max_depth = st.sidebar.number_input("Max Depth", 1, 50, 5)
    else:
        max_depth = None

else:
    algorithm = st.sidebar.selectbox(
        "Clustering Algorithm",
        ["K-Means Clustering", "Agglomerative Clustering"]
    )
    test_size = None
    max_depth = None


st.title("Machine Learning Platform")
st.subheader(f" {algorithm}")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    df = df.drop(columns=[c for c in ["Name", "Cabin", "Ticket"] if c in df.columns],
                 errors="ignore")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    labelencoder = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = labelencoder.fit_transform(df[col].astype(str))

    st.write("Processed Data")
    st.dataframe(df)

    feature_cols = st.multiselect("Select Feature Columns (X)", df.columns)

    if model_type != "Clustering":
        label_col = st.selectbox("Select Label Column (Y)", df.columns)
    else:
        label_col = None
        n_clusters = st.number_input("Clusters", 1, 20, 3)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        run_model = st.button("Run Model")

    if run_model:

        if len(feature_cols) < 1:
            st.error("Select at least one feature!")
            st.stop()

        if model_type != "Clustering" and label_col in feature_cols:
            st.error("Label column cannot be used as a feature.")
            st.stop()

        X = df[feature_cols]

        # ====================================================
        # CLUSTERING
        # ====================================================
        if model_type == "Clustering":

            if algorithm == "K-Means Clustering":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)

            clusters = model.fit_predict(X)
            df["Cluster"] = clusters

            st.success("Clustering Completed Successfully!")

            if len(feature_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8,6))
                ax.scatter(X.iloc[:,0], X.iloc[:,1], c=clusters)
                st.pyplot(fig)
            else:
                st.warning("Select at least 2 features to plot clusters.")

        # ====================================================
        # REGRESSION
        # ====================================================
        elif model_type == "Regression":

            y = df[label_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if algorithm == "Linear Regression":
                model = LinearRegression()

            elif algorithm == "Random Forest Regressor":
                model = RandomForestRegressor(max_depth=max_depth)

            elif algorithm == "Support Vector Regressor":
                model = SVR()

            elif algorithm == "Decision Tree Regressor":
                model = DecisionTreeRegressor(max_depth=max_depth)

            else:
                model = KNeighborsRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("Regression Model Trained!")
            st.write(f"### MSE: {mse}")
            st.write(f"### R² Score: {r2}")

            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            # TREE VISUALIZATION FOR DECISION TREE REGRESSOR
            if algorithm == "Decision Tree Regressor":
                dot = export_graphviz(
                    model,
                    out_file=None,
                    feature_names=feature_cols,
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                st.subheader("Decision Tree Visualization")
                st.graphviz_chart(dot)

            # FEATURE IMPORTANCE FOR RANDOM FOREST REGRESSOR
            if algorithm == "Random Forest Regressor":
                importances = model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]

                fig, ax = plt.subplots(figsize=(10,6))
                ax.bar([feature_cols[i] for i in sorted_idx], importances[sorted_idx])
                ax.set_title("Random Forest Feature Importance")
                ax.set_ylabel("Importance Score")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # ====================================================
        # CLASSIFICATION
        # ====================================================
        else:

            y = df[label_col]

            if y.dtype in ["int64", "float64"]:
                desired_bins = 4  
                unique_vals = y.nunique()

                if unique_vals > desired_bins:
                    try:
                        y_binned = pd.qcut(y, q=desired_bins, labels=list(range(desired_bins)))
                    except ValueError:
                        y_binned = pd.qcut(y, q=desired_bins, labels=list(range(desired_bins)), duplicates='drop')
                        y_binned = pd.Series(y_binned).cat.codes
                    else:
                        if hasattr(y_binned, "astype"):
                            y_binned = y_binned.astype(int)
                    y = pd.Series(y_binned, index=y.index)
                    st.info("Label converted into bins for classification.")

            if y.dtype == 'object' or str(y.dtype).startswith('category'):
                y = LabelEncoder().fit_transform(y.astype(str))
                y = pd.Series(y, index=df.index)

            if pd.Series(y).nunique() < 2:
                st.error("Label has fewer than 2 classes — cannot perform classification.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=300)

            elif algorithm == "Random Forest Classifier":
                model = RandomForestClassifier(max_depth=max_depth)

            elif algorithm == "Support Vector Classifier":
                model = SVC()

            elif algorithm == "Decision Tree Classifier":
                model = DecisionTreeClassifier(max_depth=max_depth)

            else:
                model = KNeighborsClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success("Classification Model Trained!")
            st.write(f"### Accuracy: {acc}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # DECISION TREE VISUALIZATION
            if algorithm == "Decision Tree Classifier":
                dot = export_graphviz(
                    model,
                    out_file=None,
                    feature_names=feature_cols,
                    class_names=True,
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                st.subheader("Decision Tree Visualization")
                st.graphviz_chart(dot)

            # RANDOM FOREST FEATURE IMPORTANCE (CLASSIFIER)
            if algorithm == "Random Forest Classifier":
                importances = model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]

                fig, ax = plt.subplots(figsize=(10,6))
                ax.bar([feature_cols[i] for i in sorted_idx], importances[sorted_idx])
                ax.set_title("Random Forest Feature Importance")
                ax.set_ylabel("Importance Score")
                plt.xticks(rotation=45)
                st.pyplot(fig)
