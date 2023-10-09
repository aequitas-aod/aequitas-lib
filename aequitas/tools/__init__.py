# aequitas/tools
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from functools import wraps
from typing import Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm

# ------------------- Global Parameters ------------------- 

st_e_msg='\033[91m'
nd_e_msg='\033[0m'


# ----------------------- Decorators ----------------------

""" Decorator Function: Type check (internal)

    Description:
        Check the parameters types.
        
    Parameters:
        -  library functions.
"""
def type_check(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_types = func.__annotations__

        # Check positional arguments
        for i, arg_value in enumerate(args):
            arg_name = list(arg_types.keys())[i]
            if arg_name in arg_types:
                expected_type = arg_types[arg_name]
                if (getattr(expected_type, '__origin__', None) is Union):
                    fl=True
                    for expt in expected_type.__args__:
                        if isinstance(arg_value, expt):
                            fl=False
                    if fl:
                        raise ValueError(f"{st_e_msg}Aequitas.Error: Argument '{arg_name}' must be of one of the following types: '{list(expected_type.__args__)}'. {nd_e_msg}")
                else:
                    if not isinstance(arg_value, expected_type):
                        raise ValueError(f"{st_e_msg}Aequitas.Error: Argument '{arg_name}' must be of type '{expected_type.__name__}'. {nd_e_msg}")
        
        # Check keyword arguments
        for arg_name, arg_value in kwargs.items():
            if arg_name in arg_types:
                expected_type = arg_types[arg_name]
                if (getattr(expected_type, '__origin__', None) is Union):
                    fl=True
                    for expt in expected_type.__args__:
                        if isinstance(arg_value, expt):
                            fl=False
                    if fl:
                        raise ValueError(f"{st_e_msg}Aequitas.Error: Argument '{arg_name}' must be of one of the following types: '{list(expected_type.__args__)}'. {nd_e_msg}")
                else:
                    if not isinstance(arg_value, expected_type):
                        raise ValueError(f"{st_e_msg}Aequitas.Error: Argument '{arg_name}' must be of type '{expected_type.__name__}'. {nd_e_msg}")
        
        return func(*args, **kwargs)
    
    return wrapper


# ------------------- Private Functions ------------------- 


""" Function: Sanity check attribute

    Description:
        Check if an attribute is included in the dataset
        Permorms various checks to identify discrepencies between the dataset and the given class attribute
        
    Parameters:
        - data (pd.DataFrame): A dataset.
        - attribute (str): The name of the attribute
"""
@type_check
def _check_attribute(data: pd.DataFrame, attribute: str):
    if (attribute not in data):
        raise ValueError(f"{st_e_msg}Aequitas.Error: '{str(attribute)}' is not part of the dataset.{nd_e_msg}")


""" Function: Sanity check values

    Description:
        Check if a value is included in a pool of values
        
    Parameters:
        - value_pool (list): The pool of available values.
        - values (list): A list of values.
"""
@type_check
def _check_value(value_pool: list, values: list)->list:

    # result object
    result=[]

    # iterate through the values
    for val in values:
        if val in value_pool:
            result.append(val)
    
    if (len(result)==0):
        result=value_pool

    return result


@type_check
def _check_numerical_features(data: pd.DataFrame):

    # get features
    columns = list(data.columns.values)

    # check if column data are objects
    for column in columns:
        dtype = data[column].dtype
        if (dtype=="object"):
            raise ValueError(f"{st_e_msg}Aequitas.Error: Feature: '{column}' contains text values. Please convert to numeric for the analysis.{nd_e_msg}")


# -------------------- Public Functions -------------------


""" Train a classifier using a training sample

    Description:
        Train a specified classifier using a training sample.

    Parameters:
        - training_sample (pd.DataFrame): The training sample.
        - class_attribute (str): The class attribute.
        - cl_type (str): The classifier type, Supported types: Decision_Tree, Random_Forest, K_Nearest_Neighbors, Naive_Bayes, Logistic_Regression, SVM 
        - cl_params(object): An object that specifies the classifiers scikit-learn parameters.
        - weigths (list): Sample weights. (optional)
    Returns:
        - classifier (ClassifierMixin): A trained classifier.
"""
@type_check
def train_classifier(training_sample: pd.DataFrame, class_attribute: str, ctype: str,params:dict, weights:np.ndarray = np.array([]))-> ClassifierMixin:

    X_train=pd.DataFrame([])
    Y_train=pd.Series([])

    # get training vectors
    X_train = training_sample.drop(class_attribute, axis=1)
    Y_train = (training_sample[class_attribute]).astype('int')

    # define classifier
    if (ctype=="Decision_Tree"):
        clf = DecisionTreeClassifier(**params)

    if (ctype=="Random_Forest"):
        clf = RandomForestClassifier(**params)
    
    if (ctype=="K_Nearest_Neighbors"):
        clf = KNeighborsClassifier(**params)

    if (ctype=="Naive_Bayes"):
        clf = GaussianNB(**params)

    if (ctype=="Logistic_Regression"):
        clf = LogisticRegression(**params)

    if (ctype=="SVM"):
        clf = svm.SVC(**params)
    
    # train classifier
    if ((len(weights)>0) and (ctype!="K_Nearest_Neighbors")):
        clf.fit(X_train, Y_train, sample_weight=weights)
    else:
        clf.fit(X_train, Y_train)

    return clf


""" Function: Test a clasifier using a test_sample

    Description:
        Predicts the class of a test_sample and compares it with the real values.

    Parameters:
        - classifier (ClassifierMixin): A trained classifier.
        - test (pd.Series): The test sample with the real classification values.
        - class_attribute (str):  The class attribute.
        - prediction (pd.Series): A list of the predicted values from the classifier.
        - verbose (bool): A flag indicating if results will be displayed. *optional
    Returns:
        - results (tuple): The tuple consisting of the updated predicted test_sample and  sklearn.metrics: 
          predicted_test_sample, accuracy score, confusion matrix, classification report
"""
@type_check
def test_classifier(classifier: ClassifierMixin, test_sample: pd.DataFrame, class_attribute: str, verbose: bool = False)-> tuple:

    X_test=pd.DataFrame([])
    Y_test=pd.Series([])

    # get training vectors
    X_test = test_sample.drop(class_attribute, axis=1)
    Y_test = (test_sample[class_attribute]).astype('int')
    
    # Predict on the test data
    Y_pred = classifier.predict(X_test)
    Y_pred = pd.Series(Y_pred)

    # compute classification metrics using the real and the predicted values
    accuracy = accuracy_score(Y_test, Y_pred)
    confusion = confusion_matrix(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)

    # print results
    if (verbose):
        print("Classifier Accuracy: "+"{:.2f}".format(accuracy))

    # forms new predicted test sample
    predicted_test_sample=test_sample.copy()
    predicted_test_sample[class_attribute]=Y_pred

    return predicted_test_sample, accuracy, confusion, report


""" Function: Split dataset into training and test samples

    Description:
        Splits the dataset into training and test samples.

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - ratio (float): The percentage of dataset's size that will be used as a test sample.
        - random_state (int): A integer representing the random generator seed.
    Returns:
        - result (tuple) : The training sample (pd.dataframe) and the test sample (pd.dataframe),
"""
def split_dataset(data: pd.DataFrame,ratio: float = 0.2, random_state: int =0) -> tuple:

    if (ratio>0):
        test_sample = data.sample(frac = ratio, random_state=132)
        training_sample = data.drop(test_sample.index)
    else:
        test_sample=pd.DataFrame([])
        training_sample=data


    return training_sample,test_sample


""" Function: Returns a sample from the dataset

    Description:
        Uses an index array to return a sample of the original dataset
    Parameters:
        - data (pd.DataFrame): The Dataset.
        - index (np.array): a list of indexes
    Returns:
        - new_data (pd.dataframe): The resulting sample,
"""
def get_sample_by_index(data, index):

    new_data=data.iloc[index]
    new_data.reset_index(drop=True,inplace = True)
    
    return new_data

