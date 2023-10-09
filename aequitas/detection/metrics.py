# detection/metrics.py
from typing import Union, List
import numpy as np
import pandas as pd
from aequitas.tools import type_check,_check_attribute,_check_value
from sklearn.metrics import confusion_matrix


# ------------------- Public Functions --------------------


""" Function: Compute the Combinatorial Probabilities for a sensitive attribute

    Description:
        The probability of a sensitive group to have an (positive) outcome (+):
                    |{X ∈ D : X(S) = w, X(class) = +}|
        P(S)  =   ---------------------------------- 
                          |{X ∈ D : X(S) = w}|

        where D is the dataset, S is the sensitive value {w,b} and Class is binary classification {+,-}. 
        The function works for a categorical class outcome as well.
        
    Parameters:
        - data (pd.DataFrame): The dataset containing a sensitive attribute.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - positive_outcome (str/int/float): The value of the positive outcome. (optional)
        - verbose (bool): Determines if the results will be displayed. (optional)

    Returns:
        - result (dict): A dictionary containing the probabilities results of the form:
        {
            <value of outcome> : {
                <sensitive value 1>: value of probability,
                <sensitive value 2>: value of probability,
                ...(more sensitive values)
            },
            ...(more outcomes)
        }
"""
@type_check
def stats(data: pd.DataFrame, class_attribute: str, sensitive_attribute: str, positive_outcome: Union[int, float, str] = None, verbose: bool = False) -> dict:

    # return object
    result={}

    # check validity of features
    _check_attribute(data,class_attribute)
    _check_attribute(data,sensitive_attribute)

    # combute features unique values.
    class_values=list(data[class_attribute].unique())
    sensitive_values=list(data[sensitive_attribute].unique())

    # get positive outcome value
    positive_outcomes=_check_value(class_values,[positive_outcome])

    #iterate through outcomes (class values) 
    for outcome in positive_outcomes:

        temp={}

        #iterate through sensitive values
        for value in sensitive_values:

            # compute combinatorial probability of sensitive value
            P_C=len(data[(data[sensitive_attribute] == value) & (data[class_attribute] == outcome)])        

            # compute marginal probabilitiy of sensitive value
            P=len(data[data[sensitive_attribute] == value])

            if isinstance(value,np.int64):
                value=int(value)

            # compute probability of outcome
            if (P==0):
                temp[value]=0
            else:
                temp[value]=np.abs(P_C/P)

        if isinstance(outcome,np.int64):
            outcome=int(outcome)

        result[outcome]=temp

    # print results
    if (verbose):
        print("Probabilities:")
        print_obj=pd.DataFrame.from_dict(result, orient='index')
        print(print_obj)
        print("")

    return result


""" Function: Compute the Statistical/Demographic Parity Difference

    Description:
        The Statistical/Demographic Parity Difference of a sensitive Attribute is measured using the following formula:

                    |{X ∈ D : X(S) = w, X(class) = +}|     |{X ∈ D : X(S) = b, X(class) = +}|
        SPD_s=b  =   ----------------------------------  -  ----------------------------------
                          |{X ∈ D : X(S) = w}|                  |{X ∈ D : X(S) = b}|

        where D is the dataset, S is the sensitive value {w,b} and Class is binary classification {+,-}. 
        The function works for a categorical class outcome as well.
        
    Parameters:
        - data (pd.DataFrame): The dataset containing a sensitive attribute.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - positive_outcome (str/int/float): The value of the positive outcome. (optional)
        - privileged_group (str/int/float): The value of the priviledged group. (optional)
        - verbose (bool): Determines if the results will be displayed. (optional)

    Returns:
        - result (dict): A dictionary containing the statistical parity difference results of the form:
        {
            ">50K": {
                "<privileged group 1>": {
                    <sensitive value 1>: value of statistical parity difference,
                    <sensitive value 2>: value of statistical parity difference,
                    ...(more sensitive values)
                },
                ...(more privileged groups if not specified)
            },
            ...(more outcomes if not specified)
        }
"""
@type_check
def statistical_parity(data: pd.DataFrame, class_attribute: str, sensitive_attribute: str, 
positive_outcome: Union[int, float, str] = None, privileged_group: Union[int, float, str] = None, verbose: bool = False) -> dict:

    # return object
    result={}
    metric=0

    # get probablities
    probabilities=stats(data, class_attribute, sensitive_attribute)

    #get class and sensitive values 
    class_values=list(data[class_attribute].unique())
    sensitive_values=list(data[sensitive_attribute].unique())

    # get privileged group value
    privileged_groups=_check_value(sensitive_values,[privileged_group])

    # get positive outcome value
    positive_outcomes=_check_value(class_values,[positive_outcome])

    #iterate through positive outcomes (class values)  
    for outcome in positive_outcomes:
        
        temp={}

        #iterate through privileged groups (base)
        for value1 in privileged_groups:
            base={}

            #iterate through sensitive values
            for value2 in sensitive_values:

                if isinstance(value1,np.int64):
                    value1=int(value1)

                if isinstance(value2,np.int64):
                    value2=int(value2)

                # compute statistical parity differences
                base[value2]=probabilities[outcome][value1]-probabilities[outcome][value2]

                # compute single metric if possible
                if ((len(positive_outcomes)==1) and (len(privileged_groups)==1) and (value1!=value2)):
                    metric=base[value2]

            temp[value1]=base

        if isinstance(outcome,np.int64):
            outcome=int(outcome)

        result[outcome]=temp
        result["metric"]=metric

    # print results
    if (verbose):
        print("Statistical/Demographic Parity:")
        for outcome in positive_outcomes:
            print("Outcome: ",outcome)
            print_obj=pd.DataFrame.from_dict(result[outcome], orient='index')
            print(print_obj)
            print("")
        print("")

    return result


""" Function: Compute the  Disparate Impact Ratio

    Description:
        The Disparate Impact Ratio of a sensitive Attribute is measured using the following formula:

                    |{X ∈ D : X(S) = w, X(class) = +}|      /    |{X ∈ D : X(S) = b, X(class) = +}|
        DIR_s=b  =   ----------------------------------    /     ----------------------------------
                          |{X ∈ D : X(S) = w}|            /            |{X ∈ D : X(S) = b}|

        where D is the dataset, S is the sensitive value {w,b} and Class is binary classification {+,-}. 
        The function works for a categorical class outcome as well.
        
    Parameters:
        - data (pd.DataFrame): The dataset containing a sensitive attribute.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - positive_outcome (str/int/float): The value of the positive outcome. (optional)
        - privileged_group (str/int/float): The value of the priviledged group. (optional)
        - verbose (bool): Determines if the results will be displayed. (optional)

    Returns:
        - result (dict): A dictionary containing the disparate impact ratio results of the form:
        {
            ">50K": {
                "<privileged group 1>": {
                    <sensitive value 1>: value of statistical parity difference,
                    <sensitive value 2>: value of statistical parity difference,
                    ...(more sensitive values)
                },
                ...(more privileged groups if not specified)
            },
            ...(more outcomes if not specified)
        }
"""
@type_check
def disparate_impact(data: pd.DataFrame, class_attribute: str, sensitive_attribute: str, 
positive_outcome: Union[int, float, str] = None, privileged_group: Union[int, float, str] = None, verbose: bool = False) -> dict:

    # return object
    result={}
    metric=0

    # get probablities
    probabilities=stats(data, class_attribute, sensitive_attribute)

    #get class and sensitive values 
    class_values=list(data[class_attribute].unique())
    sensitive_values=list(data[sensitive_attribute].unique())

    # get privileged group value
    privileged_groups=_check_value(sensitive_values,[privileged_group])

    # get positive outcome value
    positive_outcomes=_check_value(class_values,[positive_outcome])

    #iterate through positive outcomes (class values)  
    for outcome in positive_outcomes:

        temp={}

        #iterate through privileged groups (base)
        for value1 in privileged_groups:
            base={}

            #iterate through sensitive values
            for value2 in sensitive_values:

                if isinstance(value1,np.int64):
                    value1=int(value1)

                if isinstance(value2,np.int64):
                    value2=int(value2)

                # compute disparate impact
                base[value2]=probabilities[outcome][value2]/probabilities[outcome][value1]

                # compute single metric if possible
                if ((len(positive_outcomes)==1) and (len(privileged_groups)==1) and (value1!=value2)):
                    metric=base[value2]

            temp[value1]=base

        if isinstance(outcome,np.int64):
            outcome=int(outcome)

        result[outcome]=temp
        result["metric"]=metric

    # print results
    if (verbose):
        print("Disparate Impact:")
        for outcome in positive_outcomes:
            print("Outcome: ",outcome)
            print_obj=pd.DataFrame.from_dict(result[outcome], orient='index')
            print(print_obj)
            print("")
        print("")

    return result


""" Function: Compute confusion metrics.

    Description:
        Computes the confusion matrix of one sensitive attribute along with various metrics
        
    Parameters:
        - data (pd.DataFrame): The dataset containing the class attribute values for the specific sensitive attribute.
        - predicted_outcome (np.ndarray): An numpy array containing the predicted values.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - positive_outcome (str/int/float): The value of the positive outcome. (optional)

    Returns:
        - result (dict): a dict containing the following:
        {
            <sensitive value>:{
                -True possitives (TP)
                -True negatives (TN)
                -False positives (FP)
                -False negatives (FN)
                -True Positive Rate (TPR) / sensitivity.
                -True Negative Rate (TNR) / specificity.
                -False Positive Rate (FPR) / Type-I error.
                -False Negative Rate (FNR) / Type-II error.
                -False Discovery Rate (FDR): The ratio of the number of false positive results to the number of total positive test results.
                -False Omission Rate (FOR): The ratio of the number of individuals with a negative predicted value for which the true label is positive.
                -Positive Predictive Value (PPV): The ratio of the number of true positives to the number of true positives and false positives.
                -Negative Predictive Value (NPV): The ratio of the number of true negatives to the number of true positives and false positives.
                -Rate of Positive Predictions (RPP): Acceptance Rate
                -Rate of Negative Predictions (RNP): The ratio of the number of false and true negatives to the total observations.
                -Accuracy (ACC):  The ratio of the number of true negatives and true positives to the total observations.
            },
            ...(more sensitive values)
        }
"""
@type_check
def confusion_metrics(data: pd.DataFrame, predicted_outcome: np.ndarray, class_attribute: str, sensitive_attribute: str, 
positive_outcome: Union[int, float, str] = None, verbose: bool = False) -> dict:

    # return object
    result={}

    # Create a DataFrame with the true outcomes and the predicted outcomes
    df = data.copy()
    df['Predicted'] = predicted_outcome

    # check validity of features
    _check_attribute(data,class_attribute)
    _check_attribute(data,sensitive_attribute)

    # combute features unique values.
    class_values=list(data[class_attribute].unique())

    # check if it is binary classification
    if (len(class_values)!=2):
        raise ValueError(f"Aequitas.Error: Confusion metrics are computed for a binary class_attribute.")

    # check possitive outcome
    positive_outcome=_check_value(class_values,[positive_outcome])
    if (len(positive_outcome)==1):
        class_values.remove(positive_outcome[0])
        class_values.append(positive_outcome[0])
    class_values.sort()

    # Group the dataset by the sensitive attribute
    groups = df.groupby(sensitive_attribute)

    # iterate through sensitive values
    for group_name, group_data in groups:

        # Compute confusion matrix
        cm = confusion_matrix(group_data[class_attribute], group_data['Predicted'], labels=class_values)
        TN, FP, FN, TP = cm.ravel()

        result[group_name]={
            "TP":TP,
            "TN":TN,
            "FP":FP,
            "FN":FN,
            "TPR":TP/(TP+FN),
            "TNR":TN/(TN+FP),
            "FPR":FP/(FP+TN),
            "FNR":FN/(FN+TP),
            "FDR":FP/(FP+TP),
            "FOR":FN/(FN+TN),
            "PPV":TP/(TP+FP),
            "NPV":TN/(TN+FN),
            "RPP":(FP+TP)/(TN+TP+FN+FP),
            "RNP":(FN+TN)/(TN+TP+FN+FP),
            "ACC":(TN+TP)/(TN+TP+FN+FP),
        }

    # print results
    if (verbose):
        print(f"Confusion Metrics:  (Positive_outcome='{class_values[-1]}')")
        print_obj=pd.DataFrame.from_dict(result, orient='index')
        print(print_obj.T)

    return result


""" Function: Compute the Equality of Opportunity metric for a given sensitive variable and class attribute.

    Description:
        Computes the equality of opportunity metrics as follows:
        EoOp(S)= True possitive rate of Privileged group - True possitive rate of disadvantaged group.
        
    Parameters:
        - data (pd.DataFrame): The dataset containing a sensitive attribute.
        - predicted_outcome (np.ndarray): An numpy array containing the predicted values.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - positive_outcome (str/int/float): The value of the positive outcome. (optional)
        - privileged_group (str/int/float): The value of the priviledged group. (optional)
        - verbose (bool): Determines if the results will be displayed. (optional)

    Returns:
        - result (dict): a dict containing the Equality of Opportunity results:
        {
            "<privileged group 1>": {
                <sensitive value 1>: value of Equality of Opportunity,
                <sensitive value 2>: value of Equality of Opportunity,
                ...(more sensitive values)
            },
            ...(more privileged groups if not specified)
        }
"""
@type_check
def equal_opportunity(data: pd.DataFrame, predicted_outcome: np.ndarray, class_attribute: str, sensitive_attribute: str,
positive_outcome: Union[int, float, str] = None, privileged_group: Union[int, float, str] = None, verbose: bool = False) -> dict:

    # return object
    result={}
    metric=0

    # compute possitive outcome
    if positive_outcome==None:
        class_values=list(data[class_attribute].unique())
        class_values.sort()
        positive_outcome=int(class_values[-1])

    # get confusion metrics
    metrics=confusion_metrics(data,predicted_outcome,class_attribute,sensitive_attribute,positive_outcome=positive_outcome)

    # #get sensitive values 
    sensitive_values=list(metrics.keys())

    # get privileged group value
    privileged_groups=_check_value(sensitive_values,[privileged_group])

    #iterate through privileged groups (base)
    for value1 in privileged_groups:
        base={}
        result["analysis"]={}

        #iterate through sensitive values
        for value2 in sensitive_values:

            if isinstance(value1,np.int64):
                value1=int(value1)

            if isinstance(value2,np.int64):
                value2=int(value2)
            
            # compute equality of opportunity  metric
            base[value2]=metrics[value1]["TPR"]-metrics[value2]["TPR"]

            # compute single metric if possible
            if ((len(privileged_groups)==1) and (value1!=value2)):
                metric=base[value2]

        result["analysis"][value1]=base
    
    result["metric"]=metric
    
    # print results
    if (verbose):
        print(f"Equality of Opportunity:  (Positive_outcome='{positive_outcome}')")
        print_obj=pd.DataFrame.from_dict(result["analysis"], orient='index')
        print(print_obj)
        print("")

    return result


""" Function: Compute the Equal Odds metric for a given sensitive variable and class attribute.

    Description:
        Computes the Equal Odds metrics as follows:
        EO(S)= ((True possitive rate of Privileged group - True possitive rate of disadvantaged group) +
                (True negative rate of Privileged group - True negative rate of disadvantaged group))/2
        
    Parameters:
        - data (pd.DataFrame): The dataset containing a sensitive attribute.
        - predicted_outcome (np.ndarray): An numpy array containing the predicted values.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - positive_outcome (str/int/float): The value of the positive outcome. (optional)
        - privileged_group (str/int/float): The value of the priviledged group. (optional)
        - verbose (bool): Determines if the results will be displayed. (optional)

    Returns:
        - result (dict): a dict containing the Equal Odds results:
        {
            "<privileged group 1>": {
                <sensitive value 1>: value of Equal Odds,
                <sensitive value 2>: value of Equal Odds,
                ...(more sensitive values)
            },
            ...(more privileged groups if not specified)
        }
"""
@type_check
def equal_odds(data: pd.DataFrame, predicted_outcome: np.ndarray, class_attribute: str, sensitive_attribute: str,
positive_outcome: Union[int, float, str] = None, privileged_group: Union[int, float, str] = None, verbose: bool = False) -> dict:

    # return object
    result={}
    metric=0

    # compute possitive outcome
    if positive_outcome==None:
        class_values=list(data[class_attribute].unique())
        class_values.sort()
        positive_outcome=int(class_values[-1])

    # get confusion metrics
    metrics=confusion_metrics(data,predicted_outcome,class_attribute,sensitive_attribute,positive_outcome=positive_outcome)

    # #get sensitive values 
    sensitive_values=list(metrics.keys())

    # get privileged group value
    privileged_groups=_check_value(sensitive_values,[privileged_group])

    #iterate through privileged groups (base)
    for value1 in privileged_groups:
        base={}
        result["analysis"]={}

        #iterate through sensitive values
        for value2 in sensitive_values:

            if isinstance(value1,np.int64):
                value1=int(value1)

            if isinstance(value2,np.int64):
                value2=int(value2)

            # compute equal odds metric
            base[value2]=((metrics[value1]["TPR"]-metrics[value2]["TPR"])+(metrics[value1]["TNR"]-metrics[value2]["TNR"]))/2.0

            # compute single metric if possible
            if ((len(privileged_groups)==1) and (value1!=value2)):
                metric=base[value2]

        result["analysis"][value1]=base
    result["metric"]=metric

    # print results
    if (verbose):
        print(f"Equal Odds:  (Positive_outcome='{positive_outcome}')")
        print_obj=pd.DataFrame.from_dict(result["analysis"], orient='index')
        print(print_obj)
        print("")

    return result

