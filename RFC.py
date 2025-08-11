import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import json
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# M = Maligen (Malignant) - 1 -tumor je rakast in se lahko širi na druga tkiva
# B = Benigen (Benign) - 0 -tumor ni rakast in se ne širi


def save_logs(data):
    
    entry_with_time = {
        "timestamp": datetime.now().isoformat(),
        "data" : data
    }

    log_file = "logs.json"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    # Append new entry
    logs.append(entry_with_time)
    
    # Save updated logs
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    

def visualize_correlations(df):

    correlations = df.corr()["diagnosis"].abs()

    # 60% of the worst correlation columns will be dropped for better visualiation
    sorted_correlations = correlations.sort_values()
    num_cols_to_drop = int(0.6 * len(df.columns))
    cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index
    df_dropped = df.drop(cols_to_drop, axis=1) 
    

    plt.figure(figsize=(15, 10))
    sns.heatmap(df_dropped.corr(), annot=True, cmap="coolwarm")
    plt.show()

def show_decision_tree(prediction, test_y):

    conf_matrix = confusion_matrix(test_y, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Benign', 'Malignant'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Decision Tree Confusion Matrix')
    plt.show()


def print_stats(forest, test_X, show_decision, test_y, printt):

    if printt:
        print("Most important features: ")
        importances = dict(zip(forest.feature_names_in_, forest.feature_importances_))
        importances = {k: v for k,v in sorted(importances.items(), key=lambda x: x[1], reverse=True) }

        print("\t", end="")
        count = 0
        for key, value in importances.items():
            if value* 100 < 1.5: break
            
            print(f"{key}: {round(100*value, 2)}%", end=", ")
            count+=1
            if count % 2 == 0: 
                print()
                print("\t", end="")

        print()

    prediction = forest.predict(test_X)
    print(classification_report(test_y, prediction, digits=4))
    

    if show_decision: show_decision_tree(prediction, test_y)

    return classification_report(test_y, prediction, digits=4, output_dict=True)



# Reading csv, and ignoring column "Unnamed: 32"
df = pd.read_csv("Breast_cancer_dataset.csv", usecols=lambda col: not col.startswith("Unnamed"))

# Clearing data in DataFrame
df.dropna()
df.drop_duplicates()
df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x=='M' else 0)


# Making extra features for better preformance
df["compact_concave"] = df["compactness_mean"] * df["concavity_mean"]
df["points_per_unit_perimeter"] = df["concave points_mean"] / df["perimeter_mean"]
df["circular"] = df["perimeter_mean"] / df["radius_mean"]

df["worst_shape_score"] = df["compactness_worst"] * df["concavity_worst"]
df["worst_concave_perimeter"] = df["concave points_worst"] / df["perimeter_worst"]

df["radius_worst_ratio"] = df["radius_worst"] / df["radius_mean"]
df["area_worst_ratio"] = df["area_worst"] / df["area_mean"]
df["perimeter_worst_ratio"] = df["perimeter_worst"] / df["perimeter_mean"]


def start():

    print()
    # visualize_correlations(df)

    # TEST: 20%, TRAINING: 80%
    train_df, test_df = train_test_split(df, test_size=0.20)

    train_X_not_scaled = train_df.drop("diagnosis", axis=1)
    train_y = train_df["diagnosis"]

    test_X_not_scaled = test_df.drop("diagnosis", axis=1)
    test_y = test_df["diagnosis"]


    # Scale but keep DataFrame format, for feature importances
    scaler = StandardScaler()
    train_X = pd.DataFrame(
        scaler.fit_transform(train_X_not_scaled),
        columns=train_X_not_scaled.columns
    )

    test_X = pd.DataFrame(
        scaler.transform(test_X_not_scaled),
        columns=test_X_not_scaled.columns
    )




    # RFC Model
    forest = RandomForestClassifier(criterion="entropy")
    forest.fit(train_X, train_y)

    default_rfc_score = forest.score(test_X, test_y)
    print("Default RFC score:", default_rfc_score)

    metricsRFC = print_stats(forest, test_X, show_decision=False, test_y=test_y, printt=False)
    print("_____________________________________________________________________________________")
    print()

    # GRID SEARCH
    print("Preforming Grid Search")
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    param = {
        "n_estimators": range(100, 200, 25),
        "random_state": range(0, 10, 1),
    }

    grid_search = GridSearchCV(forest, param_grid=param, scoring="neg_mean_squared_error", cv=kf )
    grid_search.fit(train_X, train_y)

    best_forest = grid_search.best_estimator_

    grid_search_score = best_forest.score(test_X, test_y)
    print("Grid Search best forest score:", grid_search_score)
    print("Best parameters:", grid_search.best_params_)

    metricsGridSearch = print_stats(best_forest, test_X, show_decision=False, test_y=test_y, printt=False)


    data = {
        "Default RFC":{
            "score": default_rfc_score,
            "metrics": {
                "accuracy": round(metricsRFC["accuracy"], 2),
                "precision": round(metricsRFC["1"]["precision"], 2),
                "recall": round(metricsRFC["1"]["recall"], 2),
                "f1-score": round(metricsRFC["1"]["f1-score"], 2)
            }
        },
        "Grid Search": {
            "score": grid_search_score,
            "parameters": grid_search.best_params_,
            "metrics": {
                "accuracy": round(metricsGridSearch["accuracy"], 2),
                "precision": round(metricsGridSearch["1"]["precision"], 2),
                "recall": round(metricsGridSearch["1"]["recall"], 2),
                "f1-score": round(metricsGridSearch["1"]["f1-score"], 2)
            }
        }
    }

    save_logs(data)

    print("Saved to logs")



start()