from Dataset_v1 import Dataset_v1
from Analysis import FeatureAnalysis
from Model_v1 import Model_v1  
from log_reg import LogReg    
from dt import DT
from imb_rf import IMB_RF
from rus_boost import RUS_BOOST

if __name__ == "__main__":
   dataset = Dataset_v1("data/train_data_new.csv","data/test_data_new.csv",False)
   dataset.build_features_into_dataset()

   #analysis = FeatureAnalysis(dataset)
   #feature_ranking = analysis.select_features()
   #print(feature_ranking)

   features_for_training = [
      "n_past_podiums_last_5",
      "Qual_Position",
      "driver_encoding",
      #"ewa_driver_results",
      #"Race_Time_Encoding",
      #"n_past_race_wins",
      #"n_past_podiums",
      #"pos_gained_encoding",
      #"pos_gained_encoding_last_5",
      #"TopTeam_Red Bull Racing",
      #"pos_gained_encoding_simple",
      #"Qual_Q3_Time_Normal",
      #"TopTeam_Ferrari",
      #"TopTeam_McLaren",
      #"Sprint_Race_Position",
      #"Sprint_Qual_Position",
      #"TopTeam_Mercedes",
      #"n_past",
      #"DriverNumber",
      #"lag_SpeedST",
      #"lag_stint",
      #"lag_Sector2Time",
      #"lag_Sector1Time",
      #"Round_Number",
      #"Year",
      #"Race_Date_Code",
      #"lag_lap_time",
      #"lag_Sector3Time",
      #"Circuit_Type",
      #"Event_Type"
   ]


   dataset.set_features_for_training(features_for_training)

   models = [
      Model_v1(dataset,"RF_trn",False),
      #IMB_RF(dataset,"imb_rf",False),
      #RUS_BOOST(dataset,"rus_boost",False)
   ]

   for model in models:
      model.train()
      print(model.get_feature_importance())

      model.get_train_metrics(False)
      model.get_val_metrics(False)
      model.get_test_metrics(False)
   