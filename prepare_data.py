__author__ = "Recep YILDIRIM"

from get_data import get_data

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd 
import numpy as np

def prepare_data():
	train_set, test_set = get_data()

	actual_train_set = drop_empty_labels(train_set)
	actual_test_set = drop_empty_labels(test_set)

	extracted_train_set = actual_train_set.drop(columns=["Credit Score"])
	extracted_test_set = actual_test_set.drop(columns=["Credit Score"])

	train_labels = actual_train_set["Credit Score"].copy()
	test_labels = actual_test_set["Credit Score"].copy()

	dropped_train_set, columns = drop_columns(extracted_train_set)
	dropped_test_set = drop_columns(extracted_test_set, columns=columns, is_test_set=True)

	filled_train_set, num_imputer, cat_imputer = fill_empty_values(dropped_train_set)
	filled_test_set = fill_empty_values(dropped_test_set, num_imputer=num_imputer, cat_imputer=cat_imputer, is_test_set=True)

	categorized_train_set, encoder = categorize_object_columns(filled_train_set, range(-4, 0))
	categorized_test_set = categorize_object_columns(filled_test_set, range(-4, 0), encoder=encoder, is_test_set=True)

	final_train_set, scaler = standardization(categorized_train_set)
	final_test_set = standardization(categorized_test_set, scaler=scaler, is_test_set=True)

	X_train, X_valid, y_train, y_valid = train_test_split(final_train_set, train_labels, test_size=0.1)

	return {"X_train": X_train, "X_valid": X_valid, "X_test": final_test_set, "y_train": y_train, "y_valid": y_valid, "y_test": test_labels}
	

def drop_empty_labels(dataset):
    return dataset.drop(index=[index for index, row in enumerate(dataset["Credit Score"].isnull()) if row])


def drop_columns(dataset, is_test_set=False, columns=["Loan ID", "Customer ID"]):
    if is_test_set:
        return dataset.drop(columns=columns)
    
    dataset = dataset.drop(columns=["Loan Status"])
    
    for name in dataset.keys():
        if dataset[name].isnull().sum() >= 5000:
            columns.append(name)
    
    return dataset.drop(columns=columns), columns


def fill_empty_values(dataset, num_imputer=None, cat_imputer=None, is_test_set=False):
    num_columns, cat_columns = decompose_columns(dataset)
    
    if is_test_set:
        num_data = num_imputer.transform(dataset[num_columns])
        cat_data = cat_imputer.transform(dataset[cat_columns])
        return np.c_[num_data, cat_data]
    
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    
    num_data = num_imputer.fit_transform(dataset[num_columns])
    cat_data = cat_imputer.fit_transform(dataset[cat_columns])
    
    return np.c_[num_data, cat_data], num_imputer, cat_imputer


def decompose_columns(dataset):
    columns = dataset.keys()
    cat_columns = list()
    num_columns = list()
    
    for name in columns:
        if dataset[name].dtype == np.object:
            cat_columns.append(name)
        
        else:
            num_columns.append(name)
            
    return num_columns, cat_columns


def categorize_object_columns(dataset, columns, encoder=None, is_test_set=False):
    index = columns[0]
    
    if is_test_set:
        return np.c_[dataset[:, 0:index], encoder.transform(dataset[:, columns])]
    
    encoder = OneHotEncoder(sparse=False)
    return np.c_[dataset[:, 0:index], encoder.fit_transform(dataset[:, columns])], encoder


def standardization(dataset, scaler=None, is_test_set=False):
    if is_test_set:
        return scaler.transform(dataset)
    
    scaler = StandardScaler()
    
    return scaler.fit_transform(dataset), scaler
