#to evaluate the results of your model, whether during the training process or when predicting new reace. Refactor baseline code to accompany both 
from Dataset_v1 import Dataset_v1
from Model_v1 import Model_v1   
import Model
import Dataset 
import pandas as pd

""" WORK IN PROGRESS """

class ModelEval():
    def __init__(self, model: Model, dataset: Dataset):
        """
        Initialize ModelEval with model and dataset.
        
        Args:
            model: Trained model instance
            dataset: Dataset instance
        """
        if model.x_val is None or model.y_val is None:
            raise Exception("Model has no validation data - This class is only meant for exploration during training")

        # Get predictions for both training and validation data
        pred_trn, prob_trn = model.predict(model.x_trn)
        pred_val, prob_val = model.predict(model.x_val)

        # Get the original data indices for training and validation
        trn_indices = model.x_trn.index
        val_indices = model.x_val.index

        # Get the original data
        original_data = dataset.get_data()
        
        # Create dataframes for each split with their respective indices
        trn_df = pd.DataFrame({
            "Round_Number": original_data.loc[trn_indices, "Round_Number"],
            "Year": original_data.loc[trn_indices, "Year"],
            "Location": original_data.loc[trn_indices, "Location"],
            "Driver": original_data.loc[trn_indices, "BroadcastName"],
            "Actual": model.y_trn,
            "Predicted": pred_trn,
            "Probability": prob_trn
        })

        val_df = pd.DataFrame({
            "Round_Number": original_data.loc[val_indices, "Round_Number"],
            "Year": original_data.loc[val_indices, "Year"],
            "Location": original_data.loc[val_indices, "Location"],
            "Driver": original_data.loc[val_indices, "BroadcastName"],
            "Actual": model.y_val,
            "Predicted": pred_val,
            "Probability": prob_val
        })

        # Concatenate with proper indices
        self.pretty_df = pd.concat([trn_df, val_df], axis=0)
        self.pretty_df = self.pretty_df.sort_values(['Year','Round_Number','Probability'],ascending=[True,True,False])

    
    def get_df(self) -> pd.DataFrame:
        return self.pretty_df.copy()
