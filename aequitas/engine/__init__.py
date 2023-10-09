# aequitas/engine
import json
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
import aequitas.detection.descriptive_stats as dstats
import aequitas.detection.metrics as metrics
import aequitas.mitigation.data as technique
import aequitas.tools.data_manip as dm
import aequitas.mitigation.models as model


# ------------------- Private Functions ------------------- 


""" Function: Returns important parameters from parameters file.

    Description:
        Returns important parameters from parameters file.

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - params (dict): the parameters file.
    Returns:
        - tuple: class_attribute, sensitive_attributes, positive_value
"""
def _get_object_data(params:dict)-> tuple:

    try:
        class_attribute=params["class_attribute"]["name"]            
    except:
        class_attribute=''

    try:
        positive_value=params["class_attribute"]["positive_value"]            
    except:
        positive_value=''    

    try:
        sensitive_attributes=[]
        for attr in params["sensitive_attributes"]:
            sensitive_attributes.append(attr)
    except:
        sensitive_attributes=[]

    return class_attribute,sensitive_attributes,positive_value

# json encoder
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ------------------------ Classes ------------------------ 


""" Class: Aequitas 

    Description:
        Aequitas Class used by the context engine. It enhances a dataset with some additional fairness information

    Parameters:
        - dataset (pd.Dataframe): The dataset.
        - parameters (dict): A dictionary containing the additional information
    Returns:
        - An Aequitas object
"""
class Aequitas:


    # Public Variables
    dataset=pd.DataFrame()

    # Privates Variables
    __transformers=None


    # Constructors

    def __init__(self):
        self.parameters={}
        self.dataset=pd.DataFrame()

    """ Function: Constructor 

    Description:
        Initialize an Aequitas object using a parameters dict 

    Parameters:
        - dataset (pd.Dataframe): The dataset.
        - parameters (dict): A dictionary containing the additional information
    Returns:
        - An Aequitas object
"""
    def __init__(self, data:pd.DataFrame, params:dict = {}):

        self.parameters={}
        self.dataset=data.copy()
        self.parameters=params.copy()
        if ("Mitigation" not in self.parameters):
            self.parameters["Mitigation"]="False"


    """ Function: Copy contructor 

    Description:
        Copies the object to a new one.
    Returns:
        new Aequitas object
    """
    def copy(self):
        newObj=Aequitas(self.dataset,self.parameters)
        newObj.__transformers=self.__transformers
        return newObj


    # Helper Functions


    """ Function: Set parameters 

    Description:
        Set parameters to the Aequitas object.

    Parameters:
        - parameters (dict): A dict containing Aequitas object parameters

    """
    def set_params(self, parameters:dict):
        self.parameters=parameters


    """ Function: Set dataset 

    Description:
        Set dataset to the Aequitas object.

    Parameters:
        - dataset (pd.DataFrame): A dataset.

    """
    def set_dataset(self, dataset:pd.DataFrame):
        self.dataset=dataset


    """ Function: Display object  

    Description:
        Display parameter file of Aequitas object.

    Parameters:
    """
    def display(self):
        print("Aequitas Dataset parameters:")
        print(json.dumps(self.parameters, indent=4, cls=NpEncoder))


    """ Function: transform instructions 

    Description:
        It includes transformation instructions for the mitigation and model mitigation techniques. 
        it is optional since the user does not have to use our transform functions.

    Parameters:
        - transform_dict (dict): A dictionary containing the additional transformation instructions:
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
    """
    def transform_instructions(self,transform_dict:dict):
        self.parameters["transform_dictionary"]=transform_dict


    """ Function: Transform dataset per parameters transform instructions

    Description:
        Transform dataset per parameters transform instructions (if dataset is not already modifies for computation)
    """
    def transform(self):
        if "transform_dictionary" in self.parameters:
            self.dataset, self.__transformers = dm.transform_training_data(self.dataset, self.parameters["transform_dictionary"])


    """ Function: Inverse transform dataset per parameters transform instructions

    Description:
        reverts transform dataset per parameters transform instructions.
    """
    def inverse_transform(self):
        if "transform_dictionary" in self.parameters:
            self.dataset = dm.inverse_transform_data(self.dataset, self.parameters["transform_dictionary"], self.__transformers)


    # Wrapper Functions


    """ Function: Computes Aequitas object structure

    Description:
        Adds info about the features names, type, content type into the parameters file. 

    Parameters:
        - verbose (bool): A flag indicating if results will be displayed. *optional
    """
    def structure(self, verbose : bool = False):
        dstats.analyse_dataset(self.dataset,verbose=verbose)


    """ Function: Compute descriptive statistics on Aequitas object

    Description:
        Adds descriptive statistic analysis on the parameters file. The analysis is based on the class_attribute and sensitive attributes of the parameters file.

    Parameters:
        - verbose (bool): A flag indicating if results will be displayed. *optional
    """
    def descriptive_stats(self, verbose : bool = False):

        proportions={}
        outcome_distribution_by_group={}

        # get parameters
        class_attribute,sensitive_attributes,_= _get_object_data(self.parameters)

        # class item proportions
        if class_attribute!='':
            res=dstats.proportions(self.dataset,class_attribute,verbose=verbose)
            proportions[class_attribute]=res
    
        # sensitive items proportions
        if (len(sensitive_attributes)>0):
            for attr in sensitive_attributes:
                res=dstats.proportions(self.dataset,attr["name"],verbose=verbose)
                proportions[attr["name"]]=res

        # update parameter file
        self.parameters["proportions"]=proportions

        # check if class item or sensitive values exist
        if ((class_attribute=='') or ( not len(sensitive_attributes)>0)):
            return

        # get proportions per outcome
        sens_attr=[]
        for attr in sensitive_attributes:
            sens_attr.append(attr["name"])
            res=dstats.outcome_distribution_by_group(self.dataset, class_attribute, attr["name"],verbose=verbose)
            outcome_distribution_by_group[class_attribute+'/'+attr["name"]]=res
        
        # update parameter file
        self.parameters["outcome_distribution_by_group"]=outcome_distribution_by_group

        # contigency between attributes
        res=dstats.contingency(self.dataset, class_attribute, sens_attr,verbose=verbose)

        # update parameter file
        self.parameters["contingency"]=res

        # mutual information between attributes
        res=dstats.mutual_information(self.dataset, class_attribute, sens_attr,verbose=verbose)

        # update parameter file
        self.parameters["mutual_information"]=res


    """ Function: Compute statistical parity on Aequitas object

    Description:
        Adds statistical parity metric on the parameters file. The analysis is based on the class_attribute and sensitive attributes of the parameters file.

    Parameters:
        - verbose (bool): A flag indicating if results will be displayed. *optional
    """
    def statistical_parity(self, verbose : bool = False):

        # return object
        result={}

        # get parameters
        class_attribute,sensitive_attributes,positive_value=_get_object_data(self.parameters)

        if ((class_attribute=='') or ( not len(sensitive_attributes)>0)):
            print("Aequites.Error: 'statistical parity' requires the definition of a class attribute and a sensitive_attribute.")
            return

        # iterate through sensitive attributes
        for item in sensitive_attributes:
            
            # compute probabilities of outcome
            res=metrics.stats(self.dataset,class_attribute,item["name"],positive_outcome=positive_value,verbose=verbose)
            
            # update parameter file
            result["probabilities"]=res
            result["metric"]={}

            # compute metric
            if "privileged_group" in item:
                res=metrics.statistical_parity(self.dataset,class_attribute,item["name"],positive_outcome=positive_value,privileged_group=item["privileged_group"],verbose=verbose)                    
            else:
                res=metrics.statistical_parity(self.dataset,class_attribute,item["name"],positive_outcome=positive_value,verbose=verbose)        
            
            # update parameter file
            result["metric"][item["name"]]=res

        self.parameters["statistical_parity"]=result

    
    """ Function: Compute disparate impact on Aequitas object

    Description:
        Adds disparate impact metric on the parameters file. The analysis is based on the class_attribute and sensitive attributes of the parameters file.

    Parameters:
        - verbose (bool): A flag indicating if results will be displayed. *optional
    """
    def disparate_impact(self, verbose : bool = False):

        # return object
        result={}
        
        # get parameters
        class_attribute,sensitive_attributes,positive_value=_get_object_data(self.parameters)

        if ((class_attribute=='') or ( not len(sensitive_attributes)>0)):
            print("Aequites.Error: 'disparate impact' requires the definition of a class attribute and a sensitive_attribute.")
            return

        # iterate through sensitive attributes
        for item in sensitive_attributes:
            
            # compute probabilities of outcome
            res=metrics.stats(self.dataset,class_attribute,item["name"],positive_outcome=positive_value,verbose=verbose)
            
            # update parameter file
            result["probabilities"]=res
            result["metric"]={}

            # compute metric
            if "privileged_group" in item:
                res=metrics.disparate_impact(self.dataset,class_attribute,item["name"],positive_outcome=positive_value,privileged_group=item["privileged_group"],verbose=verbose)                    
            else:
                res=metrics.disparate_impact(self.dataset,class_attribute,item["name"],positive_outcome=positive_value,verbose=verbose)        
            
            # update parameter file
            result["metric"][item["name"]]=res

        self.parameters["disparate_impact"]=result
    

    """ Function: Compute equal opportunity on Aequitas object

    Description:
        Adds equal opportunity metric on the parameters file. The analysis is based on the class_attribute and sensitive attributes of the parameters file.

    Parameters:
        - prediction (np.ndarray): A numpy array that contains the predicted values
        - verbose (bool): A flag indicating if results will be displayed. *optional
    """
    def equal_opportunity(self, prediction: np.ndarray, verbose : bool = False):

        # return object
        result={}

        # get parameters
        class_attribute,sensitive_attributes,positive_value=_get_object_data(self.parameters)

        if ((class_attribute=='') or ( not len(sensitive_attributes)>0)):
            print("Aequites.Error: 'equal opportunity' requires the definition of a class attribute and a sensitive_attribute.")
            return

        # iterate through sensitive attributes
        for item in sensitive_attributes:
            
            # compute confusion metrics
            res=metrics.confusion_metrics(self.dataset,prediction,class_attribute,item["name"],positive_outcome=positive_value,verbose=verbose)
            
            # update parameter file
            result["probabilities"]=res
            result["metric"]={}

            # compute metric
            if "privileged_group" in item:
                res=metrics.equal_opportunity(self.dataset,prediction,class_attribute,item["name"],positive_outcome=positive_value,privileged_group=item["privileged_group"], verbose=True)
            else:
                res=metrics.equal_opportunity(self.dataset,prediction,class_attribute,item["name"],positive_outcome=positive_value, verbose=True)
            
            # update parameter file
            result["metric"][item["name"]]=res

        self.parameters["equal_opportunity"]=result


    """ Function: Compute equal odds on Aequitas object

    Description:
        Adds equal opportunity odds on the parameters file. The analysis is based on the class_attribute and sensitive attributes of the parameters file.

    Parameters:
        - prediction (np.ndarray): A numpy array that contains the predicted values
        - verbose (bool): A flag indicating if results will be displayed. *optional
    """
    def equal_odds(self, prediction: np.ndarray, verbose : bool = False):

        # return object
        result={}

        # get parameters
        class_attribute,sensitive_attributes,positive_value=_get_object_data(self.parameters)

        if ((class_attribute=='') or ( not len(sensitive_attributes)>0)):
            print("Aequites.Error: 'equal odds' requires the definition of a class attribute and a sensitive_attribute.")
            return

        # iterate through sensitive attributes
        for item in sensitive_attributes:
            
            # compute confusion metrics
            res=metrics.confusion_metrics(self.dataset,prediction,class_attribute,item["name"],positive_outcome=positive_value,verbose=verbose)
            
            # update parameter file
            result["confusion_metrics"]=res
            result["metric"]={}

            # compute metric
            if "privileged_group" in item:
                res=metrics.equal_odds(self.dataset,prediction,class_attribute,item["name"],positive_outcome=positive_value,privileged_group=item["privileged_group"], verbose=True)
            else:
                res=metrics.equal_odds(self.dataset,prediction,class_attribute,item["name"],positive_outcome=positive_value, verbose=True)
            
            # update parameter file
            result["metric"][item["name"]]=res

        self.parameters["equal_odds"]=result

    
    """ Function: Mitigates bias on the sensitive value

    Description:
        Mitigates bias on the sensitive value (specified on arguments or parameters file) using a method. 
        Requires transformation of the dataset to numerical values.

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - method (str): Data mitigation method.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - ranker (str):  Supported types: Decision_Tree, Random_Forest, K_Nearest_Neighbors, Naive_Bayes, Logistic_Regression, SVM, 
        - ranker_params(disc): An dictionary that specifies the classifiers scikit-learn parameters
    Returns:
        -Aequitas Object: new Aequitas with mitigated bias on sensitive value
    """
    def mitigation(self, method: str,sensitive_attribute: str ='', ranker: str = "Naive_Bayes",ranker_params: dict ={}):
        
        # get parameters
        class_attribute,sensitive_attributes,_=_get_object_data(self.parameters)

        # check if class item or sensitive values exist
        if ((class_attribute=='') or ( not len(sensitive_attributes)>0)):
            print("Aequites.Error: 'Mitigation techniques' requires the definition of a class attribute and a sensitive_attribute.")
            return

        # check if function arguments specify a sensitive attribute
        if (sensitive_attribute == ''):

            # check if dataset parameters have only one sensitive attribute
            if (len(sensitive_attributes)!=1):
                print("Aequites.Error: More than one sensitive attribute exist inside the parameters file.")
                print("Please specify which attribute to use using the 'sensitive_attribute' argument.")
                return
            else:
                sensitive_attribute=sensitive_attributes[0]["name"]

        # check if the sensitive parameter is in included in the dataset parameters
        sens_names=[]
        for item in sensitive_attributes:
            sens_names.append(item["name"])
        if (sensitive_attribute not in sens_names):
            print("Aequites.Error: The provided 'sensitive attribute' does not exist inside the parameters file of the dataset.")
            return

        # define new object
        newobj=self.copy()

        # check if the user has defined a transformation for the dataset
        newobj.transform()

        # init new parameters file
        parameters={}
        parameters["class_attribute"]=self.parameters["class_attribute"]
        parameters["sensitive_attributes"]=[]
        
        # locate sensitive attribute inside parameters file        
        for item in sensitive_attributes:
            if (item["name"]==sensitive_attribute):
                refitem=item
        parameters["sensitive_attributes"].append(refitem)

        # apply mitigation technique
        if (method=='massaging'):
            newobj.dataset=technique.massaging(newobj.dataset,class_attribute,sensitive_attribute,ranker,ranker_params)
            parameters["Mitigation"]="True"
            parameters["Mitigation_technique"]="massaging"
        
        if (method=='uniform_sampling'):
            newobj.dataset=technique.uniform_sampling(newobj.dataset,class_attribute,sensitive_attribute)
            parameters["Mitigation"]="True"
            parameters["Mitigation_technique"]="uniform_sampling"

        if (method=='preferential_sampling'):
            newobj.dataset=technique.preferential_sampling(newobj.dataset,class_attribute,sensitive_attribute,ranker,ranker_params)
            parameters["Mitigation"]="True"
            parameters["Mitigation_technique"]="preferential_sampling"

        # check if the user has defines a transformation for the dataset
        newobj.inverse_transform()

        # update parameters
        newobj.parameters=parameters

        # update transform_dictionary/transformers on new instance
        if "transform_dictionary" in self.parameters:
            newobj.parameters["transform_dictionary"]=self.parameters["transform_dictionary"]
            newobj.__transformers=self.__transformers

        return newobj
    

    """ Function: Trains a bias mitigated classifier on the sensitive value.

    Description:
        Trains a bias mitigated classifier on the sensitive value (specified on arguments or parameters file) using a method. 
        Requires transformation of the dataset to numerical values.

        Class Possitive value = 1
        Sensitive Provileged group = 1

    Parameters:
        - method (str): Data mitigation method.
        - sensitive_attribute (str): The name of the sensitive attribute.
        - classifier (str):  Supported types: Decision_Tree, Random_Forest, K_Nearest_Neighbors, Naive_Bayes, Logistic_Regression, SVM, 
        - classifier_params(disc): An dictionary that specifies the classifiers scikit-learn parameters
    Returns:
        -Aequitas Object: new Aequitas with mitigated bias on sensitive value
    """
    def mitigation_model(self, method: str,sensitive_attribute: str ='', classifier: str = "Naive_Bayes",classifier_params: dict ={})->ClassifierMixin:

        # get parameters
        class_attribute,sensitive_attributes,_=_get_object_data(self.parameters)

        # check if class item or sensitive values exist
        if ((class_attribute=='') or ( not len(sensitive_attributes)>0)):
            print("Aequites.Error: 'Mitigation techniques' requires the definition of a class attribute and a sensitive_attribute.")
            return

        # check if function arguments specify a sensitive attribute
        if (sensitive_attribute == ''):

            # check if dataset parameters have only one sensitive attribute
            if (len(sensitive_attributes)!=1):
                print("Aequites.Error: More than one sensitive attribute exist inside the parameters file.")
                print("Please specify which attribute to use using the 'sensitive_attribute' argument.")
                return
            else:
                sensitive_attribute=sensitive_attributes[0]["name"]

        # check if the sensitive parameter is in included in the dataset parameters
        sens_names=[]
        for item in sensitive_attributes:
            sens_names.append(item["name"])
        if (sensitive_attribute not in sens_names):
            print("Aequites.Error: The provided 'sensitive attribute' does not exist inside the parameters file of the dataset.")
            return

        # check if the user has defines a transformation for the dataset
        self.transform()
        
        # locate sensitive attribute inside parameters file        
        for item in sensitive_attributes:
            if (item["name"]==sensitive_attribute):
                refitem=item

        # train an unbiased classifier
        clf=None
        if (method=='re-weighting'):
            clf=model.reweighting(self.dataset,class_attribute,sensitive_attribute,classifier,classifier_params)
        
        self.inverse_transform()

        return clf