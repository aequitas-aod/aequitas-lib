# mitigation/data.py
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
from aequitas.tools import type_check
import aequitas.tools as tools
import aequitas.detection.metrics as metrics


# ------------------- Public Functions --------------------


""" Function: Returns ranking list of dataset indexes based on a classifier (ranker)

    Description:
        Applies classification on the entire dataset and computed the probabilities of each entry being classified to the corresponding class.
        Then uses these probabibilities to rank the indexes of the datatset. (Ascending Order)

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - class_attibute (str): The name of the class attribute.
        - ranker (str):  Supported types: Decision_Tree, Random_Forest, K_Nearest_Neighbors, Naive_Bayes, Logistic_Regression, SVM 
        - ranker_params(disc): An dictionary that specifies the classifiers scikit-learn parameters
    Returns:
        - ranking indexes (pd.ndarray): A nunpy array that provided the indexes of the dataset based on ascending probability order
"""
@type_check
def rank_data(data: pd.DataFrame,class_attribute: str,ranker: str = "Naive_Bayes",ranker_params: dict ={}) -> np.ndarray:

    # Define a trained classifier for ranking
    clf=tools.train_classifier(data,class_attribute,ranker,ranker_params)

    # define test vector 
    X_test=pd.DataFrame([])
    X_test = data.drop(class_attribute, axis=1)

    # get probability vector from classifier prediction
    probabilities = clf.predict_proba(X_test)

    # sort probabilities
    ranking=np.argsort(probabilities[:,0])

    return ranking


""" Function: Massage data based on ranking

    Description:
        Mitigates bias by changing the class +,- of M entries. The selection of entries to change comes from a ranking procedure 
        based on the probablities that a classifier (ranker) provides.

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - class_attibute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - ranker (str):  Supported types: Decision_Tree, Random_Forest, K_Nearest_Neighbors, Naive_Bayes, Logistic_Regression, SVM, 
        - ranker_params(disc): An dictionary that specifies the classifiers scikit-learn parameters
    Returns:
        - modified data (pd.DataFrame): The modified dataset
"""
@type_check
def massaging(data: pd.DataFrame,class_attribute: str,sensitive_attribute: str,ranker: str = "Naive_Bayes",ranker_params: dict ={})->pd.DataFrame:

    modified_data=data.copy()

    # check if feature's values are converted to numbers
    tools._check_numerical_features(modified_data)

    # check validity of features
    tools._check_attribute(modified_data,class_attribute)
    tools._check_attribute(modified_data,sensitive_attribute)

    # get classifier ranking
    ranking=rank_data(modified_data,class_attribute,ranker,ranker_params)

    # define promotion and demotion groups
    promotion=[]
    demotion=[]
    for i in range(0,len(modified_data)):
        if ((modified_data[class_attribute][ranking[i]]==0) and (modified_data[sensitive_attribute][ranking[i]]==0)):
            demotion.append(ranking[i])

        if ((modified_data[class_attribute][ranking[i]]==1) and (modified_data[sensitive_attribute][ranking[i]]==1)):
            promotion.append(ranking[i])

    # reverse the order fo the promotion group
    promotion=np.flip(promotion)

    # get dicrimination metric
    discr_result=metrics.statistical_parity(modified_data,class_attribute,sensitive_attribute,positive_outcome=1,privileged_group=1)
    discrimination=discr_result["metric"]

    # Count total entries where sensitive=0
    count_sensitive_0 = len(modified_data[modified_data[sensitive_attribute] == 0])

    # Count total entries where sensitive=1
    count_sensitive_1 = len(modified_data[modified_data[sensitive_attribute] == 1])

    # total entries
    total_len=len(modified_data)

    # compute the number of modifications required M
    M=int(np.ceil((discrimination*count_sensitive_0*count_sensitive_1)/total_len))

    # Change class value of M data by promoting and demoting entries
    for i in range(0,M):
        modified_data.loc[promotion[i],class_attribute]=0
        modified_data.loc[demotion[i],class_attribute]=1

    return modified_data


""" Function: Uniform re-sampling of the dataset to remove discrimination

    Description:
        the function drops or dublicates rows of the dataset in order to make all outcome probabilities equal.

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - class_attibute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
    Returns:
        - modified_data (pd.dataframe): The resampled dataset
"""
@type_check
def uniform_sampling(data: pd.DataFrame,class_attribute: str,sensitive_attribute: str)->pd.DataFrame:

    # check if feature's values are converted to numbers
    tools._check_numerical_features(data)

    # check validity of features
    tools._check_attribute(data,class_attribute)
    tools._check_attribute(data,sensitive_attribute)

    Dlen=len(data)
    sampled_data=pd.DataFrame()
        
    # for each group compute the N and then decide about discarding or dublicating items to match the Dlen
    pidx1=[]
    pidx2=[]
    idxs=[[0,0],[0,1],[1,0],[1,1]]    
    for k in range(len(idxs)):

        S_att=len(data[(data[sensitive_attribute] == idxs[k][0])])
        C_att=len(data[(data[class_attribute] == idxs[k][1])])
        S_C=len(data[(data[sensitive_attribute] == idxs[k][0]) & (data[class_attribute] == idxs[k][1])])

        # compute metrics
        w_S_C=(S_att*C_att)/(S_C*Dlen)

        Datt=len(data[(data[sensitive_attribute] == idxs[k][0]) & (data[class_attribute] == idxs[k][1])])            
        samp_elem=int(np.round(w_S_C*Datt))

        gdata=data[(data[sensitive_attribute] == idxs[k][0]) & (data[class_attribute] == idxs[k][1])]

        newdata = gdata.sample(n = samp_elem, replace = True, random_state = 2)
        sampled_data = pd.concat([sampled_data,newdata], ignore_index=True)        

    modified_data=sampled_data.reset_index(drop=True)

    return modified_data


""" Function: Preferential re-sampling of the dataset to remove discrimination

    Description:
        The function drops or dublicates rows of the dataset in order to make all outcome probabilities equal. 
        The selection of dropped/dublicated rows comes from a ranking vector computed using a classifier (ranker)

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - class_attibute (str): The name of the class attribute.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - ranker (str):  Classification type, DecisionTree, RandomForest, kNN, NaiveBayes, SVM    ***(only applies with preferential sampling)
        - ranker_params(object): An object that specifies the classifiers parameters   ***(only applies with preferential sampling)
    Returns:
        - modified_data (pd.dataframe): The resampled dataset
"""
@type_check
def preferential_sampling(data: pd.DataFrame,class_attribute: str,sensitive_attribute: str,ranker: str = "Naive_Bayes",ranker_params: dict ={})->pd.DataFrame:

    # check if feature's values are converted to numbers
    tools._check_numerical_features(data)

    # check validity of features
    tools._check_attribute(data,class_attribute)
    tools._check_attribute(data,sensitive_attribute)

    Dlen=len(data)
    sampled_data=pd.DataFrame()

    # get classifier ranking
    ranking=rank_data(data,class_attribute,ranker,ranker_params)

    # for each group compute the N and then decide about discarding or dublicating items to match the Dlen
    idxs=[[0,0],[0,1],[1,0],[1,1]]
    for k in range(len(idxs)):

        S_att=len(data[(data[sensitive_attribute] == idxs[k][0])])
        C_att=len(data[(data[class_attribute] == idxs[k][1])])
        S_C=len(data[(data[sensitive_attribute] == idxs[k][0]) & (data[class_attribute] == idxs[k][1])])

        # compute metrics
        w_S_C=(S_att*C_att)/(S_C*Dlen)

        Datt=len(data[(data[sensitive_attribute] == idxs[k][0]) & (data[class_attribute] == idxs[k][1])])            
        samp_elem=int(np.round(w_S_C*Datt))

        gdata=data[(data[sensitive_attribute] == idxs[k][0]) & (data[class_attribute] == idxs[k][1])]

        idx=[]
        listS=(data[sensitive_attribute] == idxs[k][0]) & (data[class_attribute] == idxs[k][1])
        idx = [i for i, val in enumerate(listS) if val]

        # Create a dictionary to map elements to their positions in ranking
        element_to_position = {element: position for position, element in enumerate(ranking)}

        # Sort the second list based on the order in the ranking
        sorted_idx = sorted(idx, key=lambda x: element_to_position.get(x, float('inf')))

        if (idxs[k][1]==1):
            sorted_idx=np.flip(sorted_idx)

        if (samp_elem<Datt):
            pidx=sorted_idx[-samp_elem:]
            duplicated_rows = [data.iloc[j] for j in pidx]

            df=pd.concat(duplicated_rows, axis=1).T
            sampled_data = pd.concat([sampled_data, df], axis=0)

        else:
            pidx=list(sorted_idx)
            if (len(sorted_idx)<(samp_elem-Datt)):
                sorted_idx = np.repeat(sorted_idx, int(np.ceil((samp_elem-Datt)/len(sorted_idx))), axis=0)
            for i in range(samp_elem-Datt):
                pidx.append(sorted_idx[i])
            duplicated_rows = [data.iloc[j] for j in pidx]

            df=pd.concat(duplicated_rows, axis=1).T
            sampled_data = pd.concat([sampled_data, df], axis=0)

    modified_data=sampled_data.reset_index(drop=True)

    

    return modified_data