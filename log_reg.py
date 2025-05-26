from Dataset import Dataset
from Model import Model
from sklearn.linear_model import LogisticRegression
#import pandas as pd
#import numpy as np

class LogReg(Model):
    def __init__(self,dataset: Dataset,is_for_pred:bool,name:str="LogReg",threshold=0.5):
        super().__init__(dataset,name,is_for_pred)
        self.threshold = threshold
        self.is_for_pred = is_for_pred

    def train(self):
        self.model = LogisticRegression(class_weight="balanced")
        self.model.fit(self.x_trn,self.y_trn)

    def predict(self,x):
        prob = self.model.predict_proba(x)[:,1]
        pred = (prob > self.threshold).astype(int)
        return pred,prob



        