# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 

@author: samme
Zero-r And One-R

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Función Zero-R
def zero_r(train_data, target):
    most_common_class = train_data[target].mode()[0]
    return most_common_class

# Función One-R
def one_r(train_data, target):
    best_feature = None
    best_rule = None
    best_accuracy = 0

    for column in train_data.drop(columns=[target]):
        rule = {}
        for value in train_data[column].unique():
            most_common_class = train_data[train_data[column] == value][target].mode()[0]
            rule[value] = most_common_class

        # Predicciones sobre el conjunto de entrenamiento
        predictions = train_data[column].map(rule)
        accuracy = np.mean(predictions == train_data[target])

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_feature = column
            best_rule = rule

    return best_feature, best_rule


def run_zero_r_one_r(file_path, target_column):
   
    data = pd.read_csv(file_path)

    # Dividir los datos en 70% entrenamiento y 30% prueba
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Algoritmo Zero-R
    zero_r_prediction = zero_r(train_data, target_column)
    
    # Predicciones para el conjunto de entrenamiento (Zero-R)
    zero_r_train_predictions = [zero_r_prediction] * len(train_data)
    zero_r_train_accuracy = np.mean(zero_r_train_predictions == train_data[target_column])
    
    # Predicciones para el conjunto de prueba (Zero-R)
    zero_r_test_predictions = [zero_r_prediction] * len(test_data)
    zero_r_test_accuracy = np.mean(zero_r_test_predictions == test_data[target_column])

    # Algoritmo One-R
    one_r_feature, one_r_rule = one_r(train_data, target_column)
    
    # Predicciones para el conjunto de entrenamiento (One-R)
    one_r_train_predictions = train_data[one_r_feature].map(one_r_rule)
    one_r_train_accuracy = np.mean(one_r_train_predictions == train_data[target_column])
    
    # Predicciones para el conjunto de prueba (One-R)
    one_r_test_predictions = test_data[one_r_feature].map(one_r_rule)
    one_r_test_accuracy = np.mean(one_r_test_predictions == test_data[target_column])

   
    print("Dataset de Entrenamiento Completo:")
    print(train_data)
    print("\nDataset de Prueba Completo:")
    print(test_data)

    # Resultados de Zero-R
    print("\nZero-R:")
    print(f"  Predicción para todas las instancias: {zero_r_prediction}")
    print(f"  Precisión en el conjunto de entrenamiento: {zero_r_train_accuracy:.2f}")
    print(f"  Precisión en el conjunto de prueba: {zero_r_test_accuracy:.2f}\n")

    # Resultados de One-R
    print("One-R:")
    print(f"  Reglas generadas: {one_r_rule}")
    print(f"  Precisión en el conjunto de entrenamiento: {one_r_train_accuracy:.2f}")
    print(f"  Precisión en el conjunto de prueba: {one_r_test_accuracy:.2f}")

# Llamada a la función con el archivo 'ContactLens.csv' y la columna objetivo 'Recommended_Lenses'
run_zero_r_one_r('ContactLens.csv', 'Recommended_Lenses')


