from Model import Model
from Dataset import Dataset
from imblearn.ensemble import RUSBoostClassifier
import pandas as pd

class RUS_BOOST(Model):
    """ Random under sampling + AdaBoost """
    def __init__(self,dataset:Dataset,name:str,is_for_pred:bool,threshold=0.5):
        super().__init__(dataset,name,is_for_pred)
        self.set_model_params(
            {
                "n_estimators": None,
                "max_depth": None,
                "min_samples_split": None,
                "min_samples_leaf": None,
                "max_features": None,
                "random_state": 27,
            }
        )
        self.is_for_pred = is_for_pred

    def train(self):
        self.model = RUSBoostClassifier(
            random_state = 27,
        )
        self.model.fit(self.x_trn,self.y_trn)
        self.find_best_threshold()

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