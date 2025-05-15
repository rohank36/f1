from abc import abstractmethod
import pandas as pd

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
    def train_val_test_split(self):
        raise NotImplementedError

    def get_data(self):
        return self.data.copy(deep=True)

    def get_name(self):
        return self.name

    