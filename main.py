from Dataset_v1 import Dataset_v1
from Analysis import FeatureAnalysis
from Model_v1 import Model_v1        

if __name__ == "__main__":
    dataset = Dataset_v1("data/train_data.csv","data/test_data.csv")
    dataset.build_features_into_dataset()
    
    #analysis = FeatureAnalysis(dataset)
    #lasso_weights = analysis.lasso_feature_weights()
    #print(lasso_weights)
    #analysis.analyze_feature_separation("driver_encoding",True,True)
    #analysis.corr_heat_map()
    #analysis.plot_feature_vs_target("TopTeam_Red Bull Racing")
    #feature_ranking = analysis.select_features()
    #print(feature_ranking)
    
    """ 
    features_for_training = ['Sprint_Qual_Position', 'Sprint_Race_Position', 'Qual_Position',
       'TopTeam_Red Bull Racing', 'TopTeam_Ferrari', 'TopTeam_Mercedes',
       'TopTeam_McLaren', 'driver_encoding', 'lag_SpeedST', 'n_past','ewa_driver_results']
    dataset.set_features_for_training(features_for_training)
    
    """
    dataset.set_features_for_training(["Qual_Position","driver_encoding"])
    model = Model_v1(dataset)
    model.train()
    print(model.get_feature_importance())
    model.get_train_metrics()
    model.get_val_metrics()
