#to evaluate the results of your model, whether during the training process or when predicting new reace. Refactor baseline code to accompany both 
from Dataset_v1 import Dataset_v1
from Model_v1 import Model_v1   
import Model
import Dataset 
import pandas as pd
from sklearn.metrics import f1_score

""" WORK IN PROGRESS """

class ModelEval():
    def __init__(self, model: Model, dataset: Dataset):
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

    def get_driver_with_most_wrong_predictions(self):
        acc_per_driver = self.get_acc_per_driver()
        worst_idx = acc_per_driver['accuracy'].idxmin()
        worst_row = acc_per_driver.loc[worst_idx]
        driver = worst_row['Driver']
        driver_acc = worst_row['accuracy']
        print(f"Driver with the most wrong predictions: {driver} {driver_acc}")
        return self.get_preds_for_driver(driver)

    def get_drivers_with_most_predictions(self):
        top1_df = self.pretty_df[self.pretty_df['Predicted'] == 1]
        pred_top_count_per_driver = (
            top1_df
            .groupby('Driver')["Predicted"]
            .count()
            .reset_index(name='count')
        )
        return pred_top_count_per_driver.sort_values(by='count',ascending=False)

    def get_drivers_with_most_predictions_by_year(self, year: int):
        top1_df = self.pretty_df[(self.pretty_df['Predicted'] == 1) & (self.pretty_df['Year'] == year)]
        pred_top_count_per_driver = (
            top1_df
            .groupby('Driver')["Predicted"]
            .count()
            .reset_index(name='count')
        )
        return pred_top_count_per_driver.sort_values(by='count',ascending=False)

    def get_acc_per_driver(self):
        self.pretty_df['correct'] = ( self.pretty_df['Predicted'] == self.pretty_df['Actual'])
        acc_per_driver = (
            self.pretty_df
            .groupby('Driver')['correct']
            .mean()
            .reset_index(name='accuracy')
        )
        return acc_per_driver.sort_values(by='accuracy')

    def get_preds_for_race(self,year,round_number):
        return self.pretty_df.loc[(self.pretty_df["Round_Number"]==round_number)&(self.pretty_df["Year"]==year),:]
    
    def get_preds_for_driver(self,driver_name):
        return self.pretty_df.loc[self.pretty_df["Driver"]==driver_name,:]

    def get_f1score_per_driver(self):
        f1_per_driver = self.pretty_df.groupby('Driver').apply(
        lambda group: f1_score(
            group['Actual'], 
            group['Predicted'], 
            )
        ).reset_index(name='f1')
    
        return f1_per_driver.sort_values(by='f1', ascending=False)

    
    def get_f1score_per_race(self):
        round_to_country = (
            self.pretty_df
            .groupby('Round_Number')['Location']
            .first()
            .to_dict()
        )

        f1_per_rd = self.pretty_df.groupby('Round_Number').apply(
            lambda group: f1_score(
                group['Actual'], 
                group['Predicted'], 
            )
        ).reset_index(name='f1')

        f1_per_rd['Location'] = f1_per_rd['Round_Number'].map(round_to_country)

        return f1_per_rd.sort_values(by='f1',ascending=False)
