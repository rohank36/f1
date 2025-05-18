from Model import Model
from Dataset import Dataset
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

class DT(Model):
    def __init__(self,dataset: Dataset, name:str="DT"):
        super().__init__(dataset,name)
        self.set_model_params(
            {
                "max_depth": 12,
                "min_samples_split": 20,
                "min_samples_leaf": 15,
                "max_features": "sqrt",
                "random_state": 27,
                "class_weight": "balanced",
                #"bootstrap": False
            }
        )
        self.threshold = 0.5

    def get_feature_importance(self):
        """
        Returns feature importance ranking and prints the top features
        """
        importances = self.model.feature_importances_
        feature_names = self.x_trn.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return importance_df

    def visualize_tree(self, max_depth=None):
        """
        Visualizes the decision tree
        """
        plt.figure(figsize=(20,10))
        plot_tree(self.model, 
                 max_depth=max_depth,
                 feature_names=self.x_trn.columns,
                 filled=True,
                 rounded=True,
                 class_names=["Class 0", "Class 1"],
                 fontsize=10)
        plt.show()

    def find_best_threshold(self,plot_curves=False):
        # find best prob threshold based on precision and recall curve
        print(f"Finding best threshold...")
        probs = self.model.predict_proba(self.x_val)[:,1]
        precisions,recalls,pr_thresholds = precision_recall_curve(self.y_val,probs)
        if plot_curves:
            plt.plot(recalls,precisions)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.show()

        f1_scores = []
        valid_thresholds = []
        for p,r,t in zip(precisions[1:], recalls[1:], pr_thresholds):
            if p+r == 0:
                continue # skip 0 values to avoid div by 0 error
            f1_scores.append(2 * p * r / (p + r))
            valid_thresholds.append(t)
        best_idx = np.argmax(f1_scores)
        best_pr_threshold = valid_thresholds[best_idx]
        print("Precision:", precisions[best_idx])
        print("Recall:", recalls[best_idx])
        print("F1 score:", f1_scores[best_idx])
        print("Best PR threshold:", best_pr_threshold)
        print("\n")
    
        # find best prob threshold based on Youden's J statistic for fpr and tpr in roc curve 
        fpr,tpr,roc_thresholds = roc_curve(self.y_val,probs)
        if plot_curves:
            plt.plot(fpr,tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.show()
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_roc_threshold = roc_thresholds[best_idx]
        print("FPR:", fpr[best_idx])
        print("TPR:", tpr[best_idx])
        print("Best ROC threshold:", best_roc_threshold)
        print("\n")

        #best_threshold = (best_pr_threshold + best_roc_threshold) / 2
        best_threshold = best_pr_threshold
        print("\nBest threshold:", best_threshold)

        # adjust model threshold to improve performance
        self.set_threshold(best_threshold)

    def set_threshold(self,threshold):
        self.threshold = threshold
        print("Threshold set to:",self.threshold)
        
    def train(self):
        self.model = DecisionTreeClassifier(
            max_depth = self.model_params["max_depth"],
            min_samples_split = self.model_params["min_samples_split"],
            min_samples_leaf = self.model_params["min_samples_leaf"],
            max_features = self.model_params["max_features"],
            random_state = self.model_params["random_state"],
            class_weight = self.model_params["class_weight"],   
        )
        self.model.fit(self.x_trn,self.y_trn)
        #self.find_best_threshold()

    def predict(self,x):
        dt_pred_prob = self.model.predict_proba(x)[:,1]
        dt_pred = (dt_pred_prob > self.threshold).astype(int)
        return dt_pred,dt_pred_prob