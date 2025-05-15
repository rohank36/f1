from Dataset_v1 import Dataset_v1
from Analysis import FeatureAnalysis        

if __name__ == "__main__":
    dataset = Dataset_v1("data/train_data.csv","data/test_data.csv")
    dataset.build_features_into_dataset()
    feature_analysis = FeatureAnalysis(dataset)
    #feature_analysis.analyze_feature_separation("driver_encoding",True,True)
    #feature_analysis.corr_heat_map()
    feature_ranking = feature_analysis.select_features()
    print(feature_ranking)
    