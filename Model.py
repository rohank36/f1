from abc import abstractmethod
from Dataset import Dataset
from typing import Dict, Any
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, r2_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, average_precision_score,fbeta_score

class Model:
    def __init__(self, dataset:Dataset, name:str, is_for_pred:bool, threshold=0.5):
        self.dataset = dataset
        self.x_trn,self.y_trn,self.x_val,self.y_val,self.x_test,self.y_test = dataset.train_val_test_split(is_for_pred)
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def tune_hyperparameters(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self,x):
        raise NotImplementedError

    def get_model_params(self) -> Dict[str,Any]:
        return self.model_params
    
    def set_model_params(self,params:Dict[str,Any]) -> None:
        self.model_params = params

    def get_model_name(self) -> str:
        return self.name

    def set_threshold(self, threshold) -> None:
        self.threshold = threshold
        print("Threshold set to:",self.threshold)

    def find_best_threshold(self,plot_curves=False):
        # find best prob threshold based on precision and recall curve
        print(f"Finding best threshold...")
        if self.is_for_pred:
            probs = self.model.predict_proba(self.x_trn)[:,1]
            precisions,recalls,pr_thresholds = precision_recall_curve(self.y_trn,probs)
        else:
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
        if self.is_for_pred: fpr,tpr,roc_thresholds = roc_curve(self.y_trn,probs)
        else: fpr,tpr,roc_thresholds = roc_curve(self.y_val,probs)
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
        print("Best threshold:", best_threshold)

        # adjust model threshold to improve performance
        self.set_threshold(best_threshold)

    def write_results(self,data_type:str, results:Dict[str,float], filename:str = "model_results.csv") -> None:
        valid_data_types = ["train","val","test"]
        if data_type not in valid_data_types: raise ValueError(f"Invalid data type. Must be one of {valid_data_types}")
        
        # Get current timestamp in a readable format
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
        # Get dataset name
        dataset_name = self.dataset.get_name()
    
        # Get model parameters
        model_params = self.get_model_params()
    
        # Combine all information
        results_with_metadata = {
            'timestamp': timestamp,
            'dataset_name': dataset_name,
            'data_type': data_type,
            'features': list(self.x_trn.columns),  # Add features list
            'model_name': self.get_model_name(),
            **model_params,  # Unpack model parameters
            **results  # Unpack original results
        }
    
        # Check if file exists
        file_exists = os.path.isfile(filename)
    
        # Write to CSV
        try:
            with open(filename, 'a', newline='') as csvfile:
                # Create fieldnames with dataset name and model params first
                fieldnames = [
                    'timestamp', 
                    'dataset_name', 
                    'data_type',
                    'features',  # Add features column
                    'model_name',
                    *model_params.keys(),  # Unpack model parameter keys
                    *results.keys()  # Unpack result keys
                ]
            
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(results_with_metadata)
        except IOError as e:
            print(f"Error writing to file {filename}: {e}")
            raise
    
        print(f"Results appended to {filename}")

    def _get_metrics(self,x,y_truth) -> Dict[str,float]:
        y_pred,y_pred_prob = self.predict(x)

        accuracy = accuracy_score(y_truth, y_pred)
        r2 = r2_score(y_truth,y_pred)

        #cm = confusion_matrix(y_truth, y_pred)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.rf.classes_)
        #disp.plot()

        if y_pred_prob is not None:
            avg_precision = average_precision_score(y_truth,y_pred_prob)
            auroc = roc_auc_score(y_truth,y_pred_prob)
        recall = recall_score(y_truth,y_pred)
        precision = precision_score(y_truth,y_pred)
        f1 = f1_score(y_truth,y_pred)
        f05 = fbeta_score(y_truth,y_pred, beta=0.5, average="binary")
        
        print("\n")
        print(f"Accuracy (exact position match): {accuracy:.3f}")
        print(f"R^2: {r2:.3f}")
        print("ROC AUC:", auroc if auroc is not None else "N/A")
        print(f"Recall: {recall} --> Of all the real positives, how many did I catch")
        print(f"Precision: {precision} --> When I say it's positive, how often am I right")
        print("Average precision:", avg_precision if avg_precision is not None else "N/A")
        print(f"F1 score: {f1}")
        print(f"F0.5 score: {f05} --> Weighted metric favoring precision over recall")

        return {
            "accuracy": accuracy,
            "r2": r2,
            "auroc": auroc,
            "recall": recall,
            "precision": precision,
            "avg_precision": avg_precision,
            "f1": f1,
            "f05": f05
        }
    
    def get_train_metrics(self,write:bool = True) -> Dict[str,float]:
        results = self._get_metrics(self.x_trn,self.y_trn)
        if write: self.write_results("train",results)
        return results  
    
    def get_val_metrics(self,write:bool = True) -> Dict[str,float]:
        results = self._get_metrics(self.x_val,self.y_val)
        if write: self.write_results("val",results)
        return results
    
    def get_test_metrics(self,write:bool = True) -> Dict[str,float]:
        results = self._get_metrics(self.x_test,self.y_test)
        if write: self.write_results("test",results)
        return results