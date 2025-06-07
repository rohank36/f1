from Dataset_v1 import Dataset_v1
from Model_v1 import Model_v1  
from imb_rf import IMB_RF    
import pandas as pd
import os 

if __name__ == "__main__":
    """
        NOTE THAT BEFORE EVERY RACE YOU MUST: 
        1. Ensure the starting grid is the same as the Qual_Position feature in data/new_race_data.csv
        2. Check for rain during race. If rain then don't bet. Prob of black swan event increases too much --> too risky
        3. Don't use optimizer anymore. Best betting method is to just put equal stake on each driver so that you if you get 2/3, you can cancel our the loss.
    """
    train_data = pd.read_csv("data/train_data_new.csv")
    test_data = pd.read_csv("data/test_data_new.csv")
    train_test_df = pd.concat([train_data,test_data], axis=0, ignore_index=True)
    train_test_df.to_csv("data/temp_train_test_for_pred.csv",index=False)

    dataset = Dataset_v1("data/temp_train_test_for_pred.csv","data/new_race_data.csv",False)
    dataset.build_features_into_dataset()

    file_path = "data/temp_train_test_for_pred.csv"
    if os.path.exists(file_path):
        os.remove(file_path)

    #features_for_training = ["Qual_Position","driver_encoding","n_past_podiums_last_5"]
    features_for_training = ["Qual_Position","driver_encoding"]

    dataset.set_features_for_training(features_for_training)

    model = Model_v1(dataset,"RF_real",True)
    #model = IMB_RF(dataset,"imb_rf_real",True)

    original_df = dataset.get_data()
    original_df = original_df.loc[(original_df["Round_Number"]==9) & (original_df["Year"]==2025),:] # change the round_number to the correct one


    #model.set_model_params(...)
    model.train()
    preds,probs = model.predict(model.x_test)

    results = model.x_test.copy()
    results["preds"] = preds
    results["probs"] = probs
    merged_df = pd.concat([results,original_df[['Location','Year','Round_Number','BroadcastName']]], axis=1)
    merged_df = merged_df.sort_values(['probs'],ascending=False)
    #merged_df.to_csv("data/preds.csv",index=False)
    print(merged_df)
    
