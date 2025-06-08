from Model import Model
from Dataset import Dataset
from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd

class IMB_RF(Model):
    def __init__(self,dataset:Dataset,name:str,is_for_pred:bool,threshold=0.5):
        super().__init__(dataset,name,is_for_pred)
        self.set_model_params(
            {
                "n_estimators": 120,
                "max_depth": 12,
                "min_samples_split": 20,
                "min_samples_leaf": 15,
                "max_features": "sqrt",
                "random_state": 27,
            }
        )
        self.is_for_pred = is_for_pred

    def train(self,find_best_threshold=True):
        self.model = BalancedRandomForestClassifier(
            random_state = 27,
            n_estimators = self.model_params["n_estimators"],
            max_depth = self.model_params["max_depth"],
            min_samples_split = self.model_params["min_samples_split"],
            min_samples_leaf = self.model_params["min_samples_leaf"],
            max_features = self.model_params["max_features"],
        )
        self.model.fit(self.x_trn,self.y_trn)
        if find_best_threshold: self.find_best_threshold()

    def predict(self,x):
        imb_pred_prob = self.model.predict_proba(x)[:,1]
        imb_pred = (imb_pred_prob > self.threshold).astype(int)
        return imb_pred,imb_pred_prob

    def get_feature_importance(self):
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.x_trn.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return feature_importance_df
