from Dataset_v1 import Dataset_v1
from Analysis import FeatureAnalysis
from Model_v1 import Model_v1        

if __name__ == "__main__":
    dataset = Dataset_v1("data/train_data.csv","data/test_data.csv")
    dataset.build_features_into_dataset()
    
    feature_analysis = FeatureAnalysis(dataset)
    #feature_analysis.analyze_feature_separation("driver_encoding",True,True)
    #feature_analysis.corr_heat_map()
    feature_ranking = feature_analysis.select_features()
    print(feature_ranking)
    
    features_for_training = ['Sprint_Qual_Position', 'Sprint_Race_Position', 'Qual_Position',
       'TopTeam_Red Bull Racing', 'TopTeam_Ferrari', 'TopTeam_Mercedes',
       'TopTeam_McLaren', 'driver_encoding', 'lag_SpeedST', 'n_past']
    dataset.set_features_for_training(features_for_training)
    
    model = Model_v1(dataset)
    model.train()
    print(model.get_feature_importance())
    model.get_train_metrics()
    model.get_val_metrics()