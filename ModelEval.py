#to evaluate the results of your model, whether during the training process or when predicting new reace. Refactor baseline code to accompany both 
from Dataset_v1 import Dataset_v1
from Model_v1 import Model_v1   
import Model
import Dataset 
import pandas as pd
from sklearn.metrics import f1_score

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

        #Features for training 
        features_used = dataset.features_for_training
        if features_used is None: raise Exception("Dataset has no features set...")


        def get_df(indices,features_used,actual,pred,prob):
            features_used.extend(["Round_Number","Year","Location","BroadcastName"])
            data_dict = {}
            for feature in features_used:
                data_dict[feature] = original_data.loc[indices,feature]
            data_dict["Actual"] = actual
            data_dict["Predicted"] = pred
            data_dict["Probability"] = prob

            return pd.DataFrame(data_dict)

        # Create dataframes for each split with their respective indices
        trn_df = get_df(trn_indices,features_used,model.y_trn,pred_trn,prob_trn)
        val_df = get_df(val_indices,features_used,model.y_val,pred_val,prob_val)
        
        # Concatenate with proper indices
        self.pretty_df = pd.concat([trn_df, val_df], axis=0)
        self.pretty_df = self.pretty_df.sort_values(['Year','Round_Number','Probability'],ascending=[True,True,False])
        self.pretty_df = self.pretty_df.rename(columns={"BroadcastName": "Driver"})

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

    def get_topk_acc(self, k: int = 3) -> pd.DataFrame:
        # Group by race (year and location)
        grouped = self.pretty_df.groupby(['Year', 'Location'])
        
        # For each race, calculate accuracy
        results = []
        for (year, location), group in grouped:
            # Get actual top-k positions
            actual_topk = set(group[group['Actual'] == 1]['Driver'])
            
            # Get predicted top-k positions based on probability
            pred_topk = set(group.nlargest(k, 'Probability')['Driver'])
            
            # Calculate accuracy as proportion of overlap between predicted and actual top-k
            overlap = len(actual_topk & pred_topk)
            acc = overlap / k
            
            results.append({
                'Location': location,
                'Year': year,
                'Acc': acc,
                'n_actual': len(actual_topk) 
            })
        
        return pd.DataFrame(results)

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

if __name__ == "__main__":
    dataset = Dataset_v1("data/train_data_new.csv","data/test_data_new.csv",False)
    dataset.build_features_into_dataset()

    features_for_training = [
        #"n_past_podiums_last_5",
        "Qual_Position",
        #"ewa_driver_results",
        "driver_encoding",
        #"Race_Time_Encoding",
        #"Qual_Q3_Time_Normal",
        #"pos_gained_encoding_simple",
        #"n_past_podiums",
        #"pos_gained_encoding",
        #"TopTeam_Red Bull Racing",
        #"TopTeam_Ferrari",
        #"TopTeam_McLaren",
        #"Sprint_Race_Position",
        #"Sprint_Qual_Position",
        #"TopTeam_Mercedes"
    ]

    dataset.set_features_for_training(features_for_training)

    model = Model_v1(dataset,"RF_trn",False)
    model.train()
    model_eval = ModelEval(model,dataset)
    #topk = model_eval.get_topk_acc()
    #print(topk[topk["Year"]==2025])
    #print(topk[topk["Year"]==2024])
    #print(topk["Acc"].mean())
    #print(topk["Acc"].std())
    #print(topk[topk["Year"]==2023])
    #print(topk[topk["Year"]==2022])
    
