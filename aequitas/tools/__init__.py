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
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin


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


# ------------------------ Classes ------------------------


""" Class: Hybrid Naive Bayes Classifier 

    Description:
        A Hybrid Naive Bayes Classifier that uses categorical features as default but the user can specify normal features as well. 

    Parameters:
        - normal_features (list): A list of all the features that presumably have normal distribution
    Returns:
        - A ClassifierMixin scikit-learn object
"""
class HybridNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    

    """ Function: Contructor

    Description:
        Set parameters to the Hybrid Naive Bayes Classifier.

    Parameters:
        - normal_features (list): A list of all the features that presumably have normal distribution

    """
    def __init__(self, normal_features: list = []):

        # get normal features
        self.normal_features = normal_features

        # Initialize separate Categorical and Gaussian NB classifiers
        self.categorical_classifier = CategoricalNB()
        self.gaussian_classifier = GaussianNB()


    """ Function: Fit classifier

    Description:
        Splits the model into two Nayve bayes, one for categorical and one for normal features

    Parameters:
        - X (pd.DataFrame): The training X sample
        - Y (pd.DataFrame): The training Y sample (class attribute values)

    """
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight:list = []):

        self.features=list(X.columns)

        categorical_features=[]
        for feature in self.features:
            if feature not in self.normal_features:
                categorical_features.append(feature)
        self.categorical_features=categorical_features


        if (len(self.categorical_features)>0):

            if (len(sample_weight)>0):
                self.categorical_classifier.fit(X[self.categorical_features], y, sample_weight=sample_weight)
            else:
                self.categorical_classifier.fit(X[self.categorical_features], y)

            self.class_prior=np.exp(self.categorical_classifier.class_log_prior_).copy()
            self.classes=self.categorical_classifier.classes_.copy()

        if (len(self.normal_features)>0):

            if (len(sample_weight)>0):
                self.gaussian_classifier.fit(X[self.normal_features], y, sample_weight=sample_weight)
            else:
                self.gaussian_classifier.fit(X[self.normal_features], y)
            
            self.class_prior=self.gaussian_classifier.class_prior_.copy()
            self.classes=self.gaussian_classifier.classes_.copy()
        
        return self

    """ Function: Predict probabilities of Hybrid Naive Bayes.

    Description:
        predicts both models probabilities and combines them to one model.

    Parameters:
        - X (pd.DataFrame): The test X sample
    
    Returns:
        - probabilities (np.ndarray): The classifier probabilities of each Y sample entry

    """
    def predict_proba(self, X: pd.DataFrame)->np.ndarray:

        # check test features
        if len(X.columns)!=len(self.features):
            raise ValueError(f"Aequitas.Error: HybridNaiveBayesClassifier requires the same number of features between train and test samples.")            

        for feat in X.columns:
            if feat not in self.features:
                raise ValueError(f"Aequitas.Error: HybridNaiveBayesClassifier requires the same features between train and test samples.")

        # compute categorical probabilities
        if (len(self.categorical_features)>0):
            X_categorical=X.copy()
            X_categorical=X_categorical[self.categorical_features]

            probs_categorical = self.categorical_classifier.predict_proba(X_categorical)

            if (len(self.normal_features)==0):
                return probs_categorical

            nc=len(self.classes)
            nr=len(probs_categorical)
            for i in range(nr):
                for j in range(nc):
                    probs_categorical[i][j]=probs_categorical[i][j]/self.class_prior[j]

        # compute normal probabilities
        if (len(self.normal_features)>0):
            X_normal=X.copy()
            X_normal=X_normal[self.normal_features]

            probs_normal = self.gaussian_classifier.predict_proba(X_normal)

            if (len(self.categorical_features)==0):
                return probs_normal

            nc=len(self.classes)
            nr=len(probs_normal)
            for i in range(nr):
                for j in range(nc):
                    probs_normal[i][j]=probs_normal[i][j]/self.class_prior[j]

        # combine probabilities
        nc=len(self.classes)
        nr=len(probs_categorical)
        temp=np.zeros((nr,nc))
        for i in range(nr):
            for j in range(nc):
                temp[i][j]=self.class_prior[j]*probs_categorical[i][j]*probs_normal[i][j]

        # normalize combined probabilities
        colsums=np.sum(temp, axis=1)
        probs=np.zeros((nr,nc))
        for i in range(nr):
            for j in range(nc):
                probs[i][j]=temp[i][j]/colsums[i]
        
        return probs


    """ Function: Predict classification, Hybrid Naive Bayes.

    Description:
        Predicts based on the two Naive Bayes classifiers.

    Parameters:
        - X (pd.DataFrame): The test X sample
    
    Returns:
        - pred (np.ndarray): The prediction vector

    """
    def predict(self, X:pd.DataFrame)->np.ndarray:

        # compute class probabilities
        probabilities=self.predict_proba(X)

        # get classes
        if (len(self.categorical_features)>0):
            classes=self.categorical_classifier.classes_
        else:
            classes=self.gaussian_classifier.classes_

        # classify
        pred=classes[np.argmax(probabilities, axis=1)]

        return pred


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
        clf = HybridNaiveBayesClassifier(**params)

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

