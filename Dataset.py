from abc import abstractmethod
import pandas as pd
from typing import List

class Dataset:

    def __init__(self,name:str,trn_data_path:str,test_data_path:str):
        trn_data = pd.read_csv(trn_data_path)
        test_data  = pd.read_csv(test_data_path)
        self.data: pd.DataFrame = pd.concat([trn_data,test_data], axis=0, ignore_index=True)
        self.clean_dataset()
        self.name = name

    @abstractmethod
    def clean_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def train_val_test_split(self,is_for_pred):
        raise NotImplementedError

    @abstractmethod
    def set_features_for_training(self,selected_features:List[str]) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def build_features_into_dataset(self) -> None:
        raise NotImplementedError

    def get_data(self):
        return self.data.copy(deep=True)

    def get_name(self):
        return self.name

    