# detection/descriptive_stats.py
import pandas as pd
import numpy as np
import itertools
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from aequitas.tools import type_check,_check_attribute

# ------------------- Private Functions ------------------- 


""" Function: Cramer's V.

    Description:Computes the Cramer's V formula as follows:
        - chi2: Chi-squared statistic from the contingency table.
        - n: Total number of observations in the contingency table.
        - phi2: Phi-squared, calculated as chi2 / n.
        - r: Number of rows in the contingency table.
        - k: Number of columns in the contingency table.
        - phi2corr: Corrected Phi-squared, calculated as max(0, phi2 - ((k-1)*(r-1)) / (n-1)).
        - rcorr: Corrected number of rows, calculated as r - ((r-1)**2) / (n-1).
        - kcorr: Corrected number of columns, calculated as k - ((k-1)**2) / (n-1).
        - Cramer's V: Square root of (phi2corr / min((kcorr-1), (rcorr-1))).

    Parameters:
        - confusion_matrix (pd.DataFrame): A 2D array-like, the contingency table.

    Returns:
        - Cramer's V value (float): Cramer's V value
"""
@type_check
def _cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# ------------------- Public Functions --------------------


""" Function: Analyse dataset's features into type, content and values.

    Description:
        Computes the type, content and values of dataset's features.

    Parameters:
        - data (pd.DataFrame): The dataset containing the attribute.
        - threshold (int): An integer number that distinguishes categorical and continuous content (arbitrary). Default value 50, *optional
        - verbose (bool): A flag indicating if results will be displayed. *optional
    Returns:
        - result (list): A list of dictionaries for each feature containing the following data:
        {
            "Column Name": <feature>,
            "Data Type": <type>,
            "Column Type (suggestion)": <column type>, "Categorical/Ordinal" / "Continuous" / "Text" / "Binary"
            "Number_Values":<count of unique values>,
            "Values":<unique values>
        }
"""
@type_check
def analyse_dataset(data: pd.DataFrame, threshold:int = 50, verbose: bool = False):
    
    # return object
    result=[]

    # iterate through columns
    for column in data.columns:
        dtype = data[column].dtype
        unique_count = data[column].nunique()
        
        # case is text
        if (dtype == "object"):
            if (unique_count <= threshold):
                column_type = "Categorical/Ordinal"
                values=data[column].unique()
            else:
                column_type = "Text"
                values="-"   

        # case is number
        if (np.issubdtype(dtype, np.number)):
            if (unique_count <= threshold):
                column_type = "Categorical/Ordinal"
                values=data[column].unique()
            else:
                column_type = "Continuous"
                values="-"          

        # case is binary number/text/bool value
        if (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_) or (dtype == "object")) and data[column].nunique() == 2:
            column_type = "Binary"
            values=data[column].unique()

        if (dtype == "object"):
            dtype = "text"

        result.append({
            "Column Name": column,
            "Data Type": dtype,
            "Column Type (suggestion)": column_type,
            "Number_Values":unique_count,
            "Values":values
        })

    if (verbose):
        print("Dataset:")
        print(pd.DataFrame(result))

    return result


""" Function: compute proportions.

    Description:
        Computes the proportions or percentages of an attribute within a dataset.

    Parameters:
        - data (pd.DataFrame): The dataset containing the attribute.
        - attribute (str): The name of the attribute.
        - verbose (bool): A flag indicating if results will be displayed.
    Returns:
        - result (dict): A dictionary containing the proportion results:
        {
            <attribute value 1>: <proportions>,
            <attribute value 2>: <proportions>,
            ...(more attribute values)
        }
"""
@type_check
def proportions(data: pd.DataFrame, attribute: str,verbose: bool = False) -> dict:

    # return object
    result={}

    # check validity of features
    _check_attribute(data,attribute)

    result = data[attribute].value_counts(normalize=True).to_dict()

    if (verbose):
        print(f"Proportions: ({attribute})")
        print_obj=pd.DataFrame.from_dict(result, orient='index')
        print(print_obj)
        print("")

    return result


""" Function: outcome distribution by group.

    Description:
        Computes the proportions of positive and negative outcomes for a sensitive group.
        This helps you see if there are disparities in outcomes between different groups

    Parameters:
        - data (pd.DataFrame): The dataset containing the attribute.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attributes (str): the name of the sensitive attribute.
        - verbose (bool): A flag indicating if results will be displayed.
    Returns:
        - result (dict): A dictionary containing the outcome distribution by sensitive group as follows:
        {
            <attribute value 1>:{
                <class value 1>: proportions,
                <class value 2>: proportions,
                ...(more class values)
            },
            ...(more attribute values)
        }
"""
@type_check
def outcome_distribution_by_group(data: pd.DataFrame, class_attribute: str, sensitive_attribute: str,verbose: bool = False) -> dict:

    # return object
    result={}

    # check validity of features
    _check_attribute(data,class_attribute)
    _check_attribute(data,sensitive_attribute)

        
    temp=data.groupby(sensitive_attribute)[class_attribute].value_counts(normalize=True).to_dict()

    # iterate through values
    for label in temp:
        if label[0] not in result:
            result[label[0]]={
                label[1]:temp[label]
            }
        else:
            result[label[0]][label[1]]=temp[label]

    if (verbose):
        print("Outcome distribution by group:")
        print_obj=pd.DataFrame.from_dict(result, orient='index')
        print(print_obj)
        print("")

    return result


""" Function: contingency through chi squared test and Cramer's V.

    Description:
        This test is used to determine whether there is a significant association between two categorical variables.
        Computes Contingency Table, Calculate the Chi-squared Statistic:
        e.g. Income Cencus (gender, income)
        Computes Contingency Table:
        |           | Income >50K | Income <=50K | Total |
        |-----------|-------------|--------------|-------|
        | Male      |     A       |      B       |  A+B  |
        | Female    |     C       |      D       |  C+D  |
        | Total     |   A+C       |    B+D       |       |
        Chi-squared = Σ((Observed - Expected)^2 / Expected)
        Expected(A) = (Row Total(Male) * Column Total(Income >50K)) / Grand Total = ( (A+B) * (A+C) ) / (A+B+C+D)

    Parameters:
        - data (pd.DataFrame): The dataset containing the attribute.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attribute (lisr(str)): A list of names corresponding to the sensitive attributes.
        - alpha (float): A significance level, which represents the threshold for statistical significance. Default value is 0.05.
        - verbose (bool): A flag indicating if results will be displayed.
    Returns:
        - result (list(dict)): A list of dictionaries containing for each group:
            1) A cross-tabulation or a contingency matrix that shows the frequency of each combination.
            2) The Chi-squared Statistic
            3) The degrees of freedom for the Chi-squared test, which is equal to (number of rows - 1) * (number of columns - 1)
            4) The expected values for each cell.
            5) The p significal value
"""
@type_check
def contingency(data: pd.DataFrame, class_attribute: str, sensitive_attributes: list,alpha: float = 0.05, verbose: bool = False) -> list:

    # form list of attributes
    attributes=[]
    for attribute in sensitive_attributes:
        if not (attribute in data):
            print("Aequitas.Warning: Sensitive attribute value is not part of the data.")
            return None
        attributes.append(attribute)

    if not (class_attribute in data):
        print("Aequitas.Warning: Class attribute value is not part of the data.")
        return None
    attributes.append(class_attribute)

    #get combinations without replacement 
    one_way_combinations = list(itertools.combinations(attributes, 2))

    contingency=[]
    for comb in one_way_combinations:
        attr1=comb[0]
        attr2=comb[1]

        # Create a contingency table
        contingency_table = pd.crosstab(data[attr1], data[attr2])

        # Perform the Chi-squared test
        chi2, p, dof, expected = chi2_contingency(contingency_table)     

        # Cramers_v test
        cramer_v = _cramers_v(contingency_table)   

        # form result list
        contingency.append({
            "attribute1":attr1,
            "attribute2":attr2,
            "contingency_table":contingency_table.to_dict(),
            "chi2":chi2,
            "cramers_v":cramer_v,
            "p":p,
            "dof":dof,
            "expected":dict(enumerate(expected.flatten(), 1)),
        })

        # Print the results
        if (verbose):
            
            print(f"\nAssociation between {attr1} and {attr2}.")
            print("Contingency Table:")
            print(contingency_table)
            print("\nChi-squared statistic:", chi2)
            print("Cramer's V:", cramer_v)
            print("Degrees of Freedom:", dof)
            print("p-value:", p)

            # Check for statistical significance
            if p < alpha:
                print(f"There is a statistically significant association between {attr1} and {attr2}.")
            else:
                print(f"There is no statistically significant association between {attr1} and {attr2}.")

    return contingency


""" Function: contingency through mutual information.

    Description:
        Compute Mutual Information (MI) between two categorical variables X and Y:
        Formula:
        - Calculate the joint probability distribution P(X, Y).
        - Calculate the marginal probability distributions P(X) and P(Y).
        - Iterate through unique values of X and Y to calculate MI:
        - P(x, y): Joint probability of X=x and Y=y.
        - p_x: Marginal probability of X=x.
        - p_y: Marginal probability of Y=y.
        - MI += P(x, y) * log2(P(x, y) / (p_x * p_y)) for each (x, y) pair.

    Parameters:
        - data (pd.DataFrame): The dataset containing the attribute.
        - class_attribute (str): The name of the class attribute.
        - sensitive_attributes (lisr(str)): A list of names corresponding to the sensitive attributes.
        - verbose (bool): A flag indicating if results will be displayed.
    Returns:
        - result (list(dict)): A list of dictionaries containing for each group the mutual information score            
"""
@type_check
def mutual_information(data: pd.DataFrame, class_attribute: str, sensitive_attributes: list,verbose: bool = False) -> list:

    # form list of attributes
    attributes=[]
    for attribute in sensitive_attributes:
        if not (attribute in data):
            print("Aequitas.Warning: Sensitive attribute value is not part of the data.")
            return None
        attributes.append(attribute)

    if not (class_attribute in data):
        print("Aequitas.Warning: Class attribute value is not part of the data.")
        return None
    attributes.append(class_attribute)

    #get combinations without replacement 
    one_way_combinations = list(itertools.combinations(attributes, 2))

    mutual_information=[]
    for comb in one_way_combinations:
        attr1=comb[0]
        attr2=comb[1]

        # Calculate Mutual Information
        mi = mutual_info_score(data[attr1], data[attr2])

        # form result list
        mutual_information.append({
            "attribute1":attr1,
            "attribute2":attr2,
            "mi":mi,
        })

        # Print the result
        if (verbose):
            print(f"Mutual Information between {str(attr1)} and {str(attr2)}: "+str(mi))

    return mutual_information
