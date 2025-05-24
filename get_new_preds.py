from Dataset_v1 import Dataset_v1
from Model_v1 import Model_v1      
import pandas as pd
import os 

if __name__ == "__main__":
    train_data = pd.read_csv("data/train_data_new.csv")
    test_data = pd.read_csv("data/test_data_new.csv")
    test_data = test_data.loc[test_data["Round_Number"]<7,:] # just for testing - delete this when doing real pred
    train_test_df = pd.concat([train_data,test_data], axis=0, ignore_index=True)
    train_test_df.to_csv("data/temp_train_test_for_pred.csv",index=False)

    dataset = Dataset_v1("data/temp_train_test_for_pred.csv","data/new_race_data.csv",False)
    dataset.build_features_into_dataset()

    file_path = "data/temp_train_test_for_pred.csv"
    if os.path.exists(file_path):
        os.remove(file_path)

    features_for_training = ["Qual_Position","driver_encoding"]

    dataset.set_features_for_training(features_for_training)

    model = Model_v1(dataset,"RF_real",True)

    original_df = dataset.get_data()
    original_df = original_df.loc[(original_df["Round_Number"]==7) & (original_df["Year"]==2025),:]


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
    
