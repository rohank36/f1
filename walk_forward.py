from Dataset_v1 import Dataset_v1
from Model_v1 import Model_v1  
from imb_rf import IMB_RF
import pandas as pd


def walk_forward(features_for_training:list)->pd.DataFrame:
    train_end_date_year = 2024
    train_end_date_round_number = 24
    data = pd.concat([pd.read_csv("data/train_data_new.csv"),pd.read_csv("data/test_data_new.csv")], axis=0, ignore_index=True)
    train_data_mask = (data["Year"] < train_end_date_year) | ( (data["Year"] == train_end_date_year) & (data["Round_Number"] <= train_end_date_round_number) )
    train_data = data[train_data_mask]
    test_data = data[~train_data_mask]
    
    unique_pairs = test_data[["Year", "Round_Number"]].drop_duplicates()
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))
    print(f"{unique_pairs_list}\n")

    final = pd.DataFrame()

    dataset = Dataset_v1("data/train_data_new.csv","data/test_data_new.csv",False) 
    dataset.build_features_into_dataset()

    for i in range(len(unique_pairs_list)):
        test_year,test_rn = unique_pairs_list[i]
        #print(f"\nWalk Forward - testing on {unique_pairs_list[i]}")
    
        dataset.set_features_for_training(features_for_training)
        model = Model_v1(dataset,"RF_trn",False)

        #change train and test set
        data = dataset.get_data()
        train_data_mask = (data["Year"] < test_year) | ( (data["Year"] == test_year) & (data["Round_Number"] < test_rn) )
        trn = data[train_data_mask]
        test = data[(data["Year"]==test_year)&(data['Round_Number']==test_rn)]

        model.y_trn = trn["target"]
        model.y_test = test["target"]
        model.x_trn = trn[dataset.features_for_training]
        model.x_test = test[dataset.features_for_training]

        model.train(False)
        #print(model.get_feature_importance())
        preds,probs = model.predict(model.x_test)

        ### INCLUDE ACTUAL
        results = model.x_test.copy()
        results["preds"] = preds
        results["probs"] = probs
        results["actual"] = model.y_test
        merged_df = pd.concat([results,test[['Location','Year','Round_Number','BroadcastName']]], axis=1)
        merged_df = merged_df.sort_values(['probs'],ascending=False)

        final = pd.concat([final,merged_df],ignore_index=True)
       
    #print(final)
    return final

def wf_get_topk_acc(df, k: int = 3) -> pd.DataFrame:
        # Group by race (year and location)
        grouped = df.groupby(['Year', 'Location'])
        
        # For each race, calculate accuracy
        results = []
        for (year, location), group in grouped:
            # Get actual top-k positions
            actual_topk = set(group[group['actual'] == 1]['BroadcastName'])
            
            # Get predicted top-k positions based on probability
            pred_topk = set(group.nlargest(k, 'probs')['BroadcastName'])
            
            # Calculate accuracy as proportion of overlap between predicted and actual top-k
            overlap = len(actual_topk & pred_topk)
            acc = overlap / k
            
            results.append({
                'Location': location,
                'Year': year,
                'Acc': acc,
                'n_actual': len(actual_topk) 
            })
        
        results_df = pd.DataFrame(results)
        print(f"Avg TopK Acc:{results_df['Acc'].mean()}")
        return results_df
         
if __name__ == "__main__":
    walk_forward()
