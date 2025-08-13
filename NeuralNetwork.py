import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import json
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt



# M = Maligen (Malignant) - 1 -tumor je rakast in se lahko širi na druga tkiva
# B = Benigen (Benign) - 0 -tumor ni rakast in se ne širi


def visualize_training(history):

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

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


def print_stats(model, test_X, test_y):

    # Evaluate
    loss, accuracy = model.evaluate(test_X, test_y)
    print(f"Test Accuracy: {accuracy:.4f}")

    print()
    print("Prediction to test_X values ('Real Value:[Prediction]')")

    wrong = []

    def loop(arr, offset, real):
        for i in range(20):
            if offset >= 71: 
                if offset + i > 141: break
            elif offset < 71: 
                if offset + i > 70: break

            # It stores what was wrong
            if real[i+offset] != arr[i+offset]: wrong.append(f"{real[i+offset]}:{arr[i+offset]}")

            print(f"{real[i+offset]}:{arr[i+offset]}", end=", ")


    pred = model.predict(test_X)

    prediction = (pred > 0.5).astype("int32") 

    for i in range(0, 71, 15):
        loop(prediction, i, test_y)
        print()

    print()
    for i in range(71, 142, 15):
        loop(prediction, i, test_y)
        print()

    return accuracy, wrong


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

    # TEST: 25%, TRAINING: 75%
    df_0 = df[df["diagnosis"] == 0]
    df_1 = df[df["diagnosis"] == 1]

    # For each class (0.125 * 2 = 25%)
    number = int(0.125 * df.shape[0])

    # You get the first few examples of class 0 and class 1
    df_0_test = df_0.head(number)
    df_1_test = df_1.head(number)

    # Remove those rows from df_0 and df_1
    df_0 = df_0.drop(df_0_test.index)
    df_1 = df_1.drop(df_1_test.index)


    # Combine it back to test/train data
    test_df = pd.concat([df_0_test, df_1_test]).reset_index(drop=True)
    train_df = pd.concat([df_0, df_1]).reset_index(drop=True)


    # Train data
    train_X_not_scaled = train_df.drop("diagnosis", axis=1)
    train_y = train_df["diagnosis"]

    # Test data
    test_X_not_scaled = test_df.drop("diagnosis", axis=1)
    test_y = test_df["diagnosis"]

    # All feature values will be between 0 and 1  
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale but keep DataFrame format, for feature importances
    train_X = pd.DataFrame(
        scaler.fit_transform(train_X_not_scaled),
        columns=train_X_not_scaled.columns
    )
    test_X = pd.DataFrame(
        scaler.transform(test_X_not_scaled),
        columns=test_X_not_scaled.columns
    )


    # MODEL
    model = Sequential([
        # 38 features
        Input(shape=(train_X.shape[1],)),   
        Dense(64, activation='relu'),  
        Dropout(0.2),                                                   
        Dense(32, activation='relu'),  
        Dropout(0.1),
        Dense(16, activation='relu'),                                   
        Dropout(0.1),
        # Binary classification
        Dense(1, activation='sigmoid') 
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.004),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Patience, koliko epoch-ov čaka še po prvem slabem preformancu
    early_stop = EarlyStopping(monitor='val_loss',
                            patience=10,
                            restore_best_weights=True)

    # Train
    history = model.fit(
        train_X, train_y,
        validation_data=(test_X, test_y),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
    )

    print()
    accuracy, wrong_predictions = print_stats(model, test_X, test_y)

    visualize_training(history)


    data = {
        "Neural Network": {
            "Test Accuracy": accuracy,
            "Wrong predictions": wrong_predictions
        }
        
    }
    print()
  
    save_logs(data)

    print("Saved to logs")

start()