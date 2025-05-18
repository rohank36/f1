from Dataset_v1 import Dataset_v1
from Analysis import FeatureAnalysis
from Model_v1 import Model_v1        

if __name__ == "__main__":
   dataset = Dataset_v1("data/train_data.csv","data/test_data.csv",True)
   dataset.build_features_into_dataset()

   analysis = FeatureAnalysis(dataset)
   feature_ranking = analysis.select_features()
   print(feature_ranking)

   
   #features_for_training = ['Qual_Position','driver_encoding']
   features_for_training = [
      #"n_past_podiums_last_5",
      "Qual_Position",
      #"ewa_driver_results",
      "driver_encoding",
      "pos_gained_encoding_simple",
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
   
   model = Model_v1(dataset)
   model.train()
   print(model.get_feature_importance())
   #model.get_train_metrics()
   model.get_val_metrics()
   #model.get_test_metrics()
   