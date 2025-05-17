from Dataset_v1 import Dataset_v1
from Analysis import FeatureAnalysis
from Model_v1 import Model_v1        

if __name__ == "__main__":
   dataset = Dataset_v1("data/train_data.csv","data/test_data.csv",True)
   dataset.build_features_into_dataset()

   analysis = FeatureAnalysis(dataset)
   feature_ranking = analysis.select_features()
   print(feature_ranking)

   
   features_for_training = ['Qual_Position','driver_encoding']
   dataset.set_features_for_training(features_for_training)
   
   
   #dataset.set_features_for_training(["Qual_Position","driver_encoding"])
   model = Model_v1(dataset)
   model.train()
   print(model.get_feature_importance())
   model.get_train_metrics()
   model.get_val_metrics()
   