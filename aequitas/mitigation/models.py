# mitigation/models.py
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
from aequitas.tools import type_check
from sklearn.base import ClassifierMixin
import aequitas.tools as tools
from sklearn.base import BaseEstimator, ClassifierMixin
import aequitas.detection.metrics as metrics

# ------------------------ Classes ------------------------


""" Class: Unbiased Naive Bayes Classifier 

    Description:
        The unbiased Naive bayes Classifier changes the classification formula from 
    
                P(C,S,A1,...,An) = P(C)P(S|C)P(A1|C)...P(An|C)
            to
                P(C,S,A1,...,An) = P(S)P(C|S)P(A1|C)...P(An|C)
    
            and iteratively changes the P(C|S) in order to make the statistical parity  metric zero. The iterative procedure is trained on a training sample.

    Parameters:
        - sensitive_attribute (str):  The sensitive attribute. Must be categorical with two outcomes in order for the classifier to work properly.
        - normal_features (list): A list of all the features that presumably have normal distribution. (*optional)
        - tol (float): The terminal number for the iterative process. If discrimination is below tol, the procedure stops. (*optional)
        - perc (float): The percentage of N(S,C) reducted in each step. Higher numbers lead to faster solutions but may lead to flactuations of the end result. (*optional)
        - Nmax (int): The maximum number of iterations (*optional)

    Returns:
        - A ClassifierMixin scikit-learn object
"""
class unbiasedNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    

    """ Function: Contructor

    Description:
        Set parameters to the unbiased Naive Bayes Classifier. 

    Parameters:
        - sensitive_attribute (str):  The sensitive attribute. Must be categorical with two outcomes in order for the classifier to work properly.
        - normal_features (list): A list of all the features that presumably have normal distribution. (*optional)
        - tol (float): The terminal number for the iterative process. If discrimination is below tol, the procedure stops. (*optional)
        - perc (float): The percentage of N(S,C) reducted in each step. Higher numbers lead to faster solutions but may lead to flactuations of the end result. (*optional)
        - Nmax (int): The maximum number of iterations (*optional)

    """
    def __init__(self, sensitive_attribute: str, normal_features: list = [],tol:float=1e-2,perc:float=0.02,Nmax:int=100):

        self.sensitive_attribute = sensitive_attribute
        self.normal_features = normal_features
        self.tol=tol
        self.perc=perc
        self.Nmax=Nmax

        # Initialize classifier
        self.classifier = tools.HybridNaiveBayesClassifier(self.normal_features)


    """ Function: computes discrimination and probabilities using a given P(C|S) probability array.

    Description:
        Splits the Unbiased Naive Bayes Classifier. The fit process contains an iterative change of P(C|S) 
        in order to make the statistical parity  metric zero. The iterative procedure is trained on a training sample.
        The rest of the features are fitted using the hybrid Naive Bayes classifier.

    Parameters:
        - X (pd.DataFrame): The training X sample
        - Psc (pd.DataFrame): The P(C|S) probability array.
        - probability_non_sens (np.ndarray): The probability array for the non sensitive features.

    """
    def train_without_discr(self,X: pd.DataFrame,Psc:np.ndarray,probability_non_sens:np.ndarray):

        classes=self.classifier.classes
        nc=len(classes)

        # combine probabilities
        nr=len(probability_non_sens)
        temp=np.zeros((nr,nc))
        for i in range(nr):
            for j in range(nc):
                temp[i][j]=self.prob_sens[X[self.sensitive_attribute][i]]*Psc[X[self.sensitive_attribute][i]][j]*probability_non_sens[i][j]

        # normalize combined probabilities
        colsums=np.sum(temp, axis=1)
        probs=np.zeros((nr,nc))
        for i in range(nr):
            for j in range(nc):
                probs[i][j]=temp[i][j]/colsums[i]

        # compute predicted vector
        pred=classes[np.argmax(probs, axis=1)]

        # get dicrimination metric
        new_X=X.copy()
        new_X["class_attribute"]=pred
        discr_result=metrics.statistical_parity(new_X,"class_attribute",self.sensitive_attribute,positive_outcome=1,privileged_group=1)
        discrimination=discr_result["metric"]

        # compute positive labels assigned by classifier
        numpos_predicted=len(pred[pred==1])

        return discrimination, numpos_predicted, probs


    """ Function: Fit classifier

    Description:
        Splits the Unbiased Naive Bayes Classifier. The fit process contains an iterative change of P(C|S) 
        in order to make the statistical parity  metric zero. The iterative procedure is trained on a training sample.
        The rest of the features are fitted using the hybrid Naive Bayes classifier.

    Parameters:
        - X (pd.DataFrame): The training X sample
        - Y (pd.DataFrame): The training Y sample (class attribute values)

    """
    def fit(self, X: pd.DataFrame, y: pd.Series):

        # compute possible outcomes of sensitive feature
        ns=len(X[self.sensitive_attribute].unique())
        outcome_sens=list(X[self.sensitive_attribute].unique())
        outcome_sens.sort()
        self.outcome_sens=outcome_sens.copy()

        # check sensitive attribute
        if (self.sensitive_attribute in self.normal_features) or (ns!=2):
            raise ValueError(f"Aequitas.Error: The sensitive attribute should be categorical with two possible outcomes")

        # fit hybrid classifier with features without sensitive value
        X_non_sens = X.drop(self.sensitive_attribute, axis=1)
        self.features=list(X.columns)
        self.classifier.fit(X_non_sens, y)

        # get class prior probability
        class_prior=self.classifier.class_prior
        classes=self.classifier.classes
        nc=len(classes)

        # check class attribute
        if (nc!=2):
            raise ValueError(f"Aequitas.Error: The class attribute should be categorical with two possible outcomes")

        # get posterior probability of features without sensitive value
        probability_non_sens = self.classifier.predict_proba(X_non_sens)

        # get likelyhood of features without sensitive value
        nr=len(probability_non_sens)
        for i in range(nr):
            for j in range(nc):
                probability_non_sens[i][j]=probability_non_sens[i][j]/class_prior[j]

        # compute sensitive probability
        prob_sens=np.zeros(2)
        for i in range(ns):
            prob_sens[i]=(len(X[X[self.sensitive_attribute] == outcome_sens[i]]))/len(X[self.sensitive_attribute])
        self.prob_sens=prob_sens.copy()

        # compute N[C|S] and P[C|S] (first time)
        Nsc=np.zeros((ns,nc))
        Psc=np.zeros((ns,nc))
        for i in range(ns):
            for j in range(nc):
                Nsc[i][j]=len(X[(y == classes[j]) & (X[self.sensitive_attribute] == outcome_sens[i])])
                Psc[i][j]=Nsc[i][j]/len(X[X[self.sensitive_attribute] == self.outcome_sens[i]])

        numpos_actual=len(y[y==1])

        discrimination, numpos_predicted,_=self.train_without_discr(X,Psc,probability_non_sens)

        cnt=0
        while ((discrimination>self.tol) & (cnt<=self.Nmax)):

            if (numpos_predicted<numpos_actual):
                Nsc[0][1]=Nsc[0][1]+self.perc*Nsc[1][0]
                Nsc[0][0]=Nsc[0][0]-self.perc*Nsc[1][0]
            else:
                Nsc[1][0]=Nsc[1][0]+self.perc*Nsc[0][1]
                Nsc[1][1]=Nsc[1][1]-self.perc*Nsc[0][1]

            # compute N[C|S] and P[C|S] (iteration)
            Psc=np.zeros((ns,nc))
            for i in range(ns):
                for j in range(nc):
                    Psc[i][j]=Nsc[i][j]/len(X[X[self.sensitive_attribute] == self.outcome_sens[i]])

            discrimination, numpos_predicted,_=self.train_without_discr(X,Psc,probability_non_sens)

            if (cnt%5==0):
                print(f"Step: {cnt} Discrimination: {discrimination} ")

            cnt+=1

        print(f"Final step: {cnt} Discrimination: {discrimination} ")
        self.Psc=Psc.copy()

        return self


    """ Function: Predict probabilities of Hybrid Naive Bayes.

    Description:
        Provides the probability vector of the classifier.

    Parameters:
        - X (pd.DataFrame): The test X sample
    
    Returns:
        - probabilities (np.ndarray): The classifier probabilities of each Y sample entry

    """
    def predict_proba(self, X:pd.DataFrame)->np.ndarray:

        # check test features
        if len(X.columns)!=len(self.features):
            raise ValueError(f"Aequitas.Error: unbiasedNaiveBayesClassifier requires the same number of features between train and test samples.")            

        for feat in X.columns:
            if feat not in self.features:
                raise ValueError(f"Aequitas.Error: unbiasedNaiveBayesClassifier requires the same features between train and test samples.")

        # get class prior probability
        class_prior=self.classifier.class_prior
        classes=self.classifier.classes
        nc=len(classes)

        # get posterior probability of features without sensitive value
        X_non_sens = X.drop(self.sensitive_attribute, axis=1)
        probability_non_sens = self.classifier.predict_proba(X_non_sens)

        # get likelyhood of features without sensitive value
        nr=len(probability_non_sens)
        for i in range(nr):
            for j in range(nc):
                probability_non_sens[i][j]=probability_non_sens[i][j]/class_prior[j]
        

        discrimination, numpos_predicted, probabilities=self.train_without_discr(X,self.Psc,probability_non_sens)

        return probabilities


    """ Function: Predict classification, unbiased Naive Bayes.

    Description:
        Provides the predictio vector of the classifier.

    Parameters:
        - X (pd.DataFrame): The test X sample
    
    Returns:
        - pred (np.ndarray): The prediction vector

    """
    def predict(self, X:pd.DataFrame)->np.ndarray:

        # compute class probabilities
        probabilities=self.predict_proba(X)

        # get classes
        classes=self.classifier.classes

        # classify
        pred=classes[np.argmax(probabilities, axis=1)]

        return pred


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


""" Class: Unbiased Naive Bayes wrapper function

    Description:
        The unbiased Naive bayes Classifier changes the classification formula from 
    
                P(C,S,A1,...,An) = P(C)P(S|C)P(A1|C)...P(An|C)
            to
                P(C,S,A1,...,An) = P(S)P(C|S)P(A1|C)...P(An|C)
    
            and iteratively changes the P(C|S) in order to make the statistical parity  metric zero. The iterative procedure is trained on a training sample.

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - class_attibute (str): The name of the class attribute.
        - sensitive_attribute (str):  The sensitive attribute. Must be categorical with two outcomes in order for the classifier to work properly.
        - normal_features (list): A list of all the features that presumably have normal distribution. (*optional)
        - tol (float): The terminal number for the iterative process. If discrimination is below tol, the procedure stops. (*optional)
        - perc (float): The percentage of N(S,C) reducted in each step. Higher numbers lead to faster solutions but may lead to flactuations of the end result. (*optional)
        - Nmax (int): The maximum number of iterations (*optional)

    Returns:
        - A ClassifierMixin scikit-learn object
"""
@type_check
def unbiasedNaiveBayes(data: pd.DataFrame,class_attribute: str,sensitive_attribute: str, normal_features: list = [],tol:float=1e-2,perc:float=0.02,Nmax:int=100)-> ClassifierMixin:

    # check if feature's values are converted to numbers
    tools._check_numerical_features(data)

    # check validity of features
    tools._check_attribute(data,class_attribute)
    tools._check_attribute(data,sensitive_attribute)

    clf=unbiasedNaiveBayesClassifier(sensitive_attribute,normal_features,tol=tol,perc=perc,Nmax=Nmax)

    X_train=pd.DataFrame([])
    Y_train=pd.Series([])

    # get training vectors
    X_train = data.drop(class_attribute, axis=1)
    Y_train = (data[class_attribute]).astype('int')

    clf.fit(X_train,Y_train)

    return clf
