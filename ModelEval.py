#to evaluate the results of your model, whether during the training process or when predicting new reace. Refactor baseline code to accompany both 
from Dataset_v1 import Dataset_v1
from Model_v1 import Model_v1   
import Model
import Dataset 
import pandas as pd

class ModelEval():
    def __init__(self,model:Model,dataset:Dataset):
        y_trn = model.y_trn
        y_val = model.y_val # might be None
        y_test = model.y_test # might be None
        if y_val is None:
            y_val = pd.DataFrame()
        if y_test is None:
            y_test = pd.DataFrame()

        y = pd.concat([y_trn,y_val],axis=0, ignore_index=True)
        y = pd.concat([y,y_test],axis=0, ignore_index=True)
        pretty_df = pd.DataFrame()
        data = dataset.get_data()
        
        pretty_df["Round_Number"] = data["Round_Number"]
        pretty_df["Driver"] = data["BroadcastName"]
        pretty_df["Location"] = data["Location"] 
        

    def __init__(self,model,x,round_number,driver,country,actual):
        self.model = model
        self.x = x
        self.round_number = round_number
        self.driver = driver
        self.country = country
        self.pred,self.prob= model.predict(x)
        self.actual = actual

        self.pretty_df = self.pretty_predictions()