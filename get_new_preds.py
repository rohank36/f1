from Dataset_v1 import Dataset_v1
from Analysis import FeatureAnalysis
from Model_v1 import Model_v1      
import pandas as pd
import os 

if __name__ == "__main__":
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
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
    #model.set_model_params(...)
    model.train()
    preds,probs = model.predict(model.x_test)
