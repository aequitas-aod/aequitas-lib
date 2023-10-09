# aequitas/tools/data_manip.py
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
import pandas as pd
import numpy as np
from aequitas.tools import type_check


# -------------------- Public Functions -------------------


""" Function: Encode and scale specified on training sample

    Description:
        The function performs labeling or one-hot encoding on columns specified by a dictionary. Further, 
        it performs various scaling techniques on the values of columns specified by a dictionary.

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - transform_dict (dict): A disctionary that contain which columns to transform along with the 
        appropriate transformation. The dict structure is as follows:
        {
            <column name>: {
                "encode": 'encode_type',
                "scaling": 'scaling_type'
            },
            ...(more columns)
        }
        supported encode_type:  "labeling" / "one-hot" 
        supported scaling_type: "standard" / "min-max" / "max-abs" / "robust" / "quantile"

    Returns: (tuple) consisting of:
        - dataset (pd.DataFrame): The formatted dataset
        - transformers (dict): A dict containing the transformer objects of a column. The dict structure is as follows:
        {
            <column name>: {
                "encoder": <encode object>,
                "scaling": <scale object>,
                "labels": {
                    <value 1>:<transformed value (number),
                    <value 2>:<transformed value (number),
                    ...(more values)
                }
            },
            ...(more columns)
        }
        supported encode_objects: "LabelEncoder()" 
        supported scaling_objects: "StandardScaler()" / "MinMaxScaler()" / "MaxAbsScaler()" / "RobustScaler()" / "QuantileTransformer(output_distribution='normal')"
"""
@type_check
def transform_training_data(data: pd.DataFrame, transform_dict:dict)->tuple:

    # copy initial dataset
    transformed_data = data.copy()
    transformers = {}

    # iterate through specified columns
    for column in transform_dict:

        item=transform_dict[column]
        temp={}
        if "encode" in item:
            
            # label encoding column 
            if (item["encode"]=="labeling"):
                encoder = LabelEncoder()
                if "labels" in item:
                    encoder.classes_ = list(item["labels"].keys())
                transformed_data[column] = encoder.fit_transform(transformed_data[column])
                temp["encoder"]=encoder

            # one-hot encoding column 
            if (item["encode"]=="one-hot"):
                transformed_data = pd.get_dummies(transformed_data, columns=[column], prefix=[column])
                temp["encoder"]=column

        if "scaling" in item:

            if ("encode" in item) and (item["encode"]=="labeling") or ("encode" not in item):
                
                # scale column using various techniques
                if (item["scaling"]=="standard"):
                    scaler = StandardScaler()
                    transformed_data[column] = scaler.fit_transform(transformed_data[column].values.reshape(-1, 1))
                    temp["scaling"]=scaler

                if (item["scaling"]=="min-max"):
                    scaler = MinMaxScaler()
                    transformed_data[column] = scaler.fit_transform(transformed_data[column].values.reshape(-1, 1))
                    temp["scaling"]=scaler

                if (item["scaling"]=="max-abs"):
                    scaler = MaxAbsScaler()
                    transformed_data[column] = scaler.fit_transform(transformed_data[column].values.reshape(-1, 1))
                    temp["scaling"]=scaler

                if (item["scaling"]=="robust"):
                    scaler = RobustScaler()
                    transformed_data[column] = scaler.fit_transform(transformed_data[column].values.reshape(-1, 1))
                    temp["scaling"]=scaler

                if (item["scaling"]=="quantile"):
                    scaler = QuantileTransformer(output_distribution='normal')
                    transformed_data[column] = scaler.fit_transform(transformed_data[column].values.reshape(-1, 1))
                    temp["scaling"]=scaler

        transformers[column]=temp

    return transformed_data, transformers


""" Function: Encode and scale specified on test sample

    Description:
        The function performs labeling or one-hot encoding on columns specified by a dictionary. Further, 
        it performs various scaling techniques on the values of columns specified by a dictionary.

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - transform_dict (dict): A disctionary that contain which columns to transform along with the 
            appropriate transformation. The dict structure is as follows:
            {
                <column name>: {
                    "encode": 'encode_type',
                    "scaling": 'scaling_type'
                },
                ...(more columns)
            }
            supported encode_type:  "labeling" / "one-hot" 
            supported scaling_type: "standard" / "min-max" / "max-abs" / "robust" / "quantile"
        - transformers (dict): A dict containing the transformer objects of a column provided by the transform_training_data function. 
            The dict structure is as follows:
            {
                <column name>: {
                    "encoder": <encode object>,
                    "scaling": <scale object>
                },
                ...(more columns)
            }
            supported encode_objects: "LabelEncoder()" 
            supported scaling_objects: "StandardScaler()" / "MinMaxScaler()" / "MaxAbsScaler()" / "RobustScaler()" / "QuantileTransformer(output_distribution='normal')"
    Returns: tuple consisting of:
        - dataset (pd.DataFrame): The formatted dataset
        - transformers (dict): A dict containing the transformer objects of a column. The dict structure is as follows:
        {
            <column name>: {
                "encoder": <encode object>,
                "scaling": <scale object>
            },
            ...(more columns)
        }
        supported encode_objects: "LabelEncoder()" 
        supported scaling_objects: "StandardScaler()" / "MinMaxScaler()" / "MaxAbsScaler()" / "RobustScaler()" / "QuantileTransformer(output_distribution='normal')"
"""
@type_check
def transform_test_data(data: pd.DataFrame, transform_dict:dict, transformers:dict)->pd.DataFrame:

    # copy initial dataset
    transformed_data = data.copy()

    # iterate through specified columns
    for column in transform_dict:

        item=transform_dict[column]
        if "encode" in item:
            
            # label encoding column 
            if (item["encode"]=="labeling"):
                transformed_data[column] = transformers[column]["encoder"].transform(transformed_data[column])

            # one-hot encoding column 
            if (item["encode"]=="one-hot"):
                transformed_data = pd.get_dummies(transformed_data, columns=[column], prefix=[column])

        if "scaling" in item:

            if ("encode" in item) and (item["encode"]=="labeling") or ("encode" not in item):
                
                # scale column using various techniques
                if (item["scaling"]=="standard"):
                    transformed_data[column] = transformers[column]["scaling"].transform(transformed_data[column].values.reshape(-1, 1))

                if (item["scaling"]=="min-max"):
                    transformed_data[column] = transformers[column]["scaling"].transform(transformed_data[column].values.reshape(-1, 1))

                if (item["scaling"]=="max-abs"):
                    transformed_data[column] = transformers[column]["scaling"].transform(transformed_data[column].values.reshape(-1, 1))

                if (item["scaling"]=="robust"):
                    transformed_data[column] = transformers[column]["scaling"].transform(transformed_data[column].values.reshape(-1, 1))

                if (item["scaling"]=="quantile"):
                    transformed_data[column] = transformers[column]["scaling"].transform(transformed_data[column].values.reshape(-1, 1))


    return transformed_data


""" Function: Reverse encode and scale specified columns. *(use to reverse transform_training_data modifications)

    Description:
        The function performs reverse labeling or reverse one-hot encoding on columns specified by a dictionary. Further, 
        it performs various reverse scaling techniques on the values of columns specified by a dictionary.

    Parameters:
        - data (pd.DataFrame): The Dataset.
        - transform_dict (dict): A disctionary that contain which columns to transform along with the 
            appropriate transformation. The dictionary structure is as follows:
            {
                <column name 1>: {
                    "encode": 'encode_type',
                    "scaling": 'scaling_type'
                },
                ...(more columns)
            }
            supported encode_type:  "labeling" / "one-hot" 
            supported scaling_type: "standard" / "min-max" / "max-abs" / "robust" / "quantile"
        - transformers (dict): A dict containing the transformer objects of a column. The dict structure is as follows:
            {
                <column name>: {
                    "encoder": <encode object>,
                    "scaling": <scale object>
                },
                ...(more columns)
            }
            supported encode_objects: "LabelEncoder()" 
            supported scaling_objects: "StandardScaler()" / "MinMaxScaler()" / "MaxAbsScaler()" / "RobustScaler()" / "QuantileTransformer(output_distribution='normal')"
    Returns:
        - result (pd.DataFrame): The formatted dataset
"""
@type_check
def inverse_transform_data(data: pd.DataFrame, transform_dict:dict, transformers:dict)->pd.DataFrame:

    # copy initial dataset
    transformed_data = data.copy()

    # iterate through specified columns
    for column in transform_dict:

        item=transform_dict[column]

        # reverse scale column using various techniques
        if "scaling" in item:

            if ("encode" in item) and (item["encode"]=="labeling") or ("encode" not in item):

                if (item["scaling"]=="standard"):
                    transformed_data[column] = transformers[column]["scaling"].inverse_transform(transformed_data[column].values.reshape(-1, 1))
                
                if (item["scaling"]=="min-max"):
                    transformed_data[column] = transformers[column]["scaling"].inverse_transform(transformed_data[column].values.reshape(-1, 1))

                if (item["scaling"]=="max-abs"):
                    transformed_data[column] = transformers[column]["scaling"].inverse_transform(transformed_data[column].values.reshape(-1, 1))

                if (item["scaling"]=="robust"):
                    transformed_data[column] = transformers[column]["scaling"].inverse_transform(transformed_data[column].values.reshape(-1, 1))

                if (item["scaling"]=="quantile"):
                    transformed_data[column] = transformers[column]["scaling"].inverse_transform(transformed_data[column].values.reshape(-1, 1))

        if "encode" in item:

            # reverse label encoding column 
            if (item["encode"]=="labeling"):
                transformed_data[column] = transformers[column]["encoder"].inverse_transform(transformed_data[column].astype(int))

            # reverse one-hot encoding column 
            if (item["encode"]=="one-hot"):
                color_columns = [col for col in transformed_data.columns if col.startswith(column+"_")]
                transformed_data[column] = transformed_data[color_columns].idxmax(axis=1).str.replace(column+"_", '')
                transformed_data = transformed_data.drop(columns=color_columns)


    return transformed_data


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
@type_check
def split_dataset(data: pd.DataFrame,ratio: float = 0.2, random_state: int =0)->tuple:

    if (ratio>0):
        test_sample = data.sample(frac = ratio, random_state=132)
        training_sample = data.drop(test_sample.index)
    else:
        test_sample=pd.DataFrame([])
        training_sample=data

    training_sample.reset_index(drop=True,inplace = True)
    test_sample.reset_index(drop=True,inplace = True)

    return training_sample,test_sample


""" Function: Returns a sample from the dataset

    Description:
        Uses an index array to return a sample of the original dataset
    Parameters:
        - data (pd.DataFrame): The Dataset.
        - index (np.ndarray): A list of indexes
    Returns:
        - new_data (pd.dataframe): The resulting sample,
"""
@type_check
def get_sample_by_index(data: pd.DataFrame, index:np.ndarray)->pd.DataFrame:

    new_data=data.iloc[index]
    new_data.reset_index(drop=True,inplace = True)
    
    return new_data


""" Function: Groups feature's values

    Description:
        The functions groups values of a feature based on a list provided.
    Parameters:
        - data (pd.Series): The Dataset.
        - groups (list): A list of the new groups, i.e.:
        [
            ['Val1', 'Val2'],
            ['Val3', 'Val4', 'Val5']
        ]
        - labels (list): A list of names for the new grouped values. It should be the same size as the list groups. 
        If not then the names are a compination of the previous values. (optional)
    Returns:
        - modified_data (pd.Series): The modified column,
"""
@type_check
def merge_values(data: pd.Series,groups:list, labels: list =[])->pd.Series:
    
    modified_data=data.copy()

    # iterate through groups
    for i in range(len(groups)):

        # define value for the group
        if (len(labels)==len(groups)):
            glabel=labels[i]
        else:
            glabel=''
            for item in groups[i]:
                glabel+=str(item)+"/"

        # iterate through values of a group
        for item in groups[i]:
            modified_data[modified_data==item] = glabel

    return modified_data