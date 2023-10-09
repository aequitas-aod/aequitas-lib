# mitigation/models.py
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
from aequitas.tools import type_check
from sklearn.base import ClassifierMixin
import aequitas.tools as tools


# ------------------- Public Functions --------------------


""" Function: Compute weights for classifications that allow sample weights (no Knn classification)

    Description:
        It computes weights for each row of the dataset in order to use them to the classification process.

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - class_attibute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
    Returns:
        - weigths (np.ndarray): The computed weights
"""
@type_check
def reweighting_data(data: pd.DataFrame,class_attribute: str,sensitive_attribute: str)->np.ndarray:

    # check if feature's values are converted to numbers
    tools._check_numerical_features(data)

    # check validity of features
    tools._check_attribute(data,class_attribute)
    tools._check_attribute(data,sensitive_attribute)

    # compute metrics
    Dlen=len(data)

    S_0=len(data[(data[sensitive_attribute] == 0)])/Dlen
    S_1=len(data[(data[sensitive_attribute] == 1)])/Dlen
    C_0=len(data[(data[class_attribute] == 0)])/Dlen
    C_1=len(data[(data[class_attribute] == 1)])/Dlen

    S_0_C_0=len(data[(data[sensitive_attribute] == 0) & (data[class_attribute] == 0)])/Dlen
    S_0_C_1=len(data[(data[sensitive_attribute] == 0) & (data[class_attribute] == 1)])/Dlen
    S_1_C_0=len(data[(data[sensitive_attribute] == 1) & (data[class_attribute] == 0)])/Dlen
    S_1_C_1=len(data[(data[sensitive_attribute] == 1) & (data[class_attribute] == 1)])/Dlen

    # compute metrics
    w_0_0=round((S_0*C_0)/S_0_C_0,4)
    w_0_1=round((S_0*C_1)/S_0_C_1,4)
    w_1_0=round((S_1*C_0)/S_1_C_0,4)
    w_1_1=round((S_1*C_1)/S_1_C_1,4)

    weigths=np.zeros(Dlen)

    list_0_0=(data[sensitive_attribute] == 0) & (data[class_attribute] == 0)
    idx_0_0 = [i for i, val in enumerate(list_0_0) if val]

    list_0_1=(data[sensitive_attribute] == 0) & (data[class_attribute] == 1)
    idx_0_1 = [i for i, val in enumerate(list_0_1) if val]

    list_1_0=(data[sensitive_attribute] == 1) & (data[class_attribute] == 0)
    idx_1_0 = [i for i, val in enumerate(list_1_0) if val]

    list_1_1=(data[sensitive_attribute] == 1) & (data[class_attribute] == 1)
    idx_1_1 = [i for i, val in enumerate(list_1_1) if val]

    weigths[idx_0_0]=w_0_0
    weigths[idx_0_1]=w_0_1
    weigths[idx_1_0]=w_1_0
    weigths[idx_1_1]=w_1_1

    return weigths


""" Function: Returns a modified with weight classifier

    Description:
        It computes weights for each row of the dataset in order to use them to a modified trained classifier.

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - class_attibute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - classifier_type (str):  Supported types: Decision_Tree, Random_Forest, Naive_Bayes, Logistic_Regression, SVM 
        - classifier_params(disc): An dictionary that specifies the classifiers scikit-learn parameters
    Returns:
        - weigths (np.ndarray): The computed weights
"""
@type_check
def reweighting(data: pd.DataFrame,class_attribute: str,sensitive_attribute: str,classifier_type: str = "Naive_Bayes",classifier_params: dict ={})-> ClassifierMixin:

    # compute weights based on reweighting technique
    sa_weights=reweighting_data(data,class_attribute,sensitive_attribute)

    # Train classifier 
    clf=tools.train_classifier(data,class_attribute,classifier_type,classifier_params, weights=sa_weights)

    return clf