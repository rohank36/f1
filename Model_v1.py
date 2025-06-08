from Model import Model
from Dataset import Dataset
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import GridSearchCV, StratifiedKFold
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, r2_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier

class Model_v1(Model):
    def __init__(self,dataset: Dataset, name:str, is_for_pred:bool, threshold=0.5):
        super().__init__(dataset,name,is_for_pred)
        self.set_model_params(
            {
                "n_estimators": 120,
                "max_depth": 12,
                "min_samples_split": 20,
                "min_samples_leaf": 15,
                "max_features": "sqrt",
                "random_state": 27,
                #"class_weight": None,
                "class_weight": "balanced",
                "bootstrap": False,
                #"class_weight": "balanced_subsample",
                #"bootstrap": True
            }
        )
        self.is_for_pred = is_for_pred
    
    def tune_hyperparameters(self):
        raise NotImplementedError
        params = {
            "n_estimators": [50, 100, 120, 150, 200],  # Number of trees
            "max_depth": [5, 10, 12, 15, None],  # Maximum depth of trees
            "min_samples_split": [2, 10, 20, 30],  # Minimum number of samples required to split an internal node
            "min_samples_leaf": [1, 5, 15, 30],  # Minimum number of samples required to be at a leaf node
            "max_features": ["sqrt", "log2", None],  # Number of features to consider when looking for the best split
            "class_weight": [None, "balanced"]  # Handling class imbalance
        }
        
        # Use stratified k-fold cross-validation to ensure representative sampling
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=27),
            param_grid=params,
            cv=cv,
            scoring=['f1', 'precision', 'recall', 'roc_auc'],
            refit='f1',  # Use F1 score as the primary metric for refit
            return_train_score=True,
            verbose=1,
            n_jobs=-1  
        )
        
        grid_search.fit(self.x_val, self.y_val)
        self.tune_results = pd.DataFrame(grid_search.cv_results_)
        best_params = grid_search.best_params_
        
        # set best params
        self.set_model_params({
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "min_samples_split": best_params["min_samples_split"],
            "min_samples_leaf": best_params["min_samples_leaf"],
            "max_features": best_params["max_features"],
            "random_state": 27,
            "class_weight": best_params["class_weight"]
        })
        
        # Print out detailed results
        print("\nBest Parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"Best F1 Score: {grid_search.best_score_}")
    
        return best_params
    
    def train(self,find_best_threshold=True):
        self.model = RandomForestClassifier(
            n_estimators = self.model_params["n_estimators"],
            max_depth = self.model_params["max_depth"],
            min_samples_split = self.model_params["min_samples_split"],
            min_samples_leaf = self.model_params["min_samples_leaf"],
            max_features = self.model_params["max_features"],
            random_state = self.model_params["random_state"],
            class_weight = self.model_params["class_weight"],   
            bootstrap = self.model_params["bootstrap"]
        )
        #max_samples = 20?
        self.model.fit(self.x_trn,self.y_trn)
        if find_best_threshold: self.find_best_threshold()
    
    def predict(self,x):
        rf_pred_prob = self.model.predict_proba(x)[:,1]
        rf_pred = (rf_pred_prob > self.threshold).astype(int)
        return rf_pred,rf_pred_prob
        
    def get_feature_importance(self):
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.x_trn.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return feature_importance_df