import pandas as pd
from Dataset import Dataset
from typing import List

class Dataset_v1(Dataset):

    def __init__(self,trn_data_path:str,test_data_path:str):
        super().__init__(trn_data_path,test_data_path)

    def clean_dataset(self) -> None:
        # clean event type
        self.data["Event_Type"] = (self.data["Event_Type"] == "Race").astype(int)
        
        # clean Race NaNs
        cols_to_fill = ["Sprint_Qual_Position", "Sprint_Race_Position","Sector1Time", "Sector2Time", "Sector3Time", "SpeedST", "Stint", "Race_Position"]
        for col in self.data.columns:
            if col in cols_to_fill:
                self.data[col] = self.data[col].fillna(-1)
        
        # clean teams
        self.data.loc[(self.data["TeamName"] == "AlphaTauri") | (self.data["TeamName"] == "RB"), "TeamName"] = "Racing Bulls"
        self.data.loc[(self.data["TeamName"] == "Alfa Romeo"), "TeamName"] = "Kick Sauber"
        unique_teams =  self.data["TeamName"].unique()
        if len(unique_teams) != 10: raise Exception(f"Wrong number of teams in data, there are: {len(unique_teams)} teams")

        # clean driver names 
        # replace NaN driver broadcast names
        drivers_dict = self.data.groupby("DriverNumber")["BroadcastName"].unique().apply(list).to_dict()
        nan_driver_names = self.data.loc[self.data["BroadcastName"].isna(),:]
        for index, row in nan_driver_names.iterrows():
            self.data.loc[index,"BroadcastName"] = drivers_dict[row["DriverNumber"]][0]
        
        self.data.loc[self.data["DriverNumber"]==12,"BroadcastName"] = "K ANTONELLI"
        if len(self.data["BroadcastName"].unique()) != 31: raise Exception("Wrong number of unique driver names")
        
        self.data = self.data.reset_index(drop=True)

    def train_val_test_split(self,cols_to_drop:List[str]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        data_copy = self.data.copy(deep=True)
        data_copy.drop(columns=cols_to_drop, inplace=True)
        #data_copy.drop(columns=["Location"], inplace=True)
        #data_copy.drop(columns=["TeamName"], inplace=True)
        #data_copy.drop(columns=["Circuit_Type"], inplace=True)        
        #return train,val,test
    
    def create_top_team(self) -> None:
        self.data['TopTeam_Red Bull Racing'] = (self.data['TeamName'] == 'Red Bull Racing').astype(int)
        self.data['TopTeam_Ferrari'] = (self.data['TeamName'] == 'Ferrari').astype(int)
        self.data['TopTeam_Mercedes'] = (self.data['TeamName'] == 'Mercedes').astype(int)
        self.data['TopTeam_McLaren'] = (self.data['TeamName'] == 'McLaren').astype(int)

    def create_circuit_type(self) -> None:
        # One hot encode circuit type. 
        location_to_circuit_type = {
            'Sakhir': 'race',
            'Jeddah': 'street',
            'Melbourne': 'street',
            'Imola': 'race',
            'Miami': 'street',
            'Barcelona': 'race',
            'Monaco': 'street',
            'Baku': 'street',
            'Montréal': 'street',
            'Silverstone': 'race',
            'Spielberg': 'race',
            'Le Castellet': 'race',
            'Budapest': 'race',
            'Spa-Francorchamps': 'race',
            'Zandvoort': 'race',
            'Monza': 'race',
            'Marina Bay': 'street',
            'Suzuka': 'race',
            'Austin': 'race',
            'Mexico City': 'race',
            'São Paulo': 'race',
            'Yas Island': 'race',
            'Lusail': 'race',
            'Las Vegas': 'street',
            'Shanghai': 'race'
        }
        self.data["Circuit_Type"] = self.data["Location"].map(location_to_circuit_type)
        self.data["Circuit_Type"] = (self.data["Circuit_Type"] == "race").astype(int)

    def create_race_date_code(self) -> None:
        self.data['Race_Date_Code'] = self.data['Year']*100 + self.data['Round_Number']

    def create_driver_encoding(self) -> None:
        # Calculate Driver Encoding Feature 
        self.data = self.data.sort_values(['BroadcastName','Race_Date_Code']) 
        self.data['n_past']  = self.data.groupby('BroadcastName').cumcount() # number of past races up to t-1

        mu = self.data['Race_Position'].mean() # global mean 
        k  = 20 # smoothing parameter (defines the number of races needed to be considered not a rookie)
        alpha = 0.3 # EMA smoothing factor. means alpha% weight on the most recent race i.e. St = alpha * xt + (1-alpha) * St-1

        self.data['ema_past'] = (
            self.data
            .groupby('BroadcastName',group_keys=False)['Race_Position']
            .apply(lambda x: x.shift(1).ewm(alpha=alpha, adjust=True).mean())
        )

        # exponential weighted average of historice race positions
        # this encoding reflects both the drivers recent form and the uncertainty that comes if they've only raced a little
        self.data['driver_encoding'] = (self.data['n_past'] * self.data['ema_past'] + k  * mu) / (self.data['n_past'] + k)

        driver_encoding_ranked = (
            self.data
            .sort_values(['BroadcastName','Race_Date_Code'])
            .groupby('BroadcastName')['driver_encoding']
            .last()
            .reset_index()
        )
        driver_encoding_ranked.sort_values(by='driver_encoding', inplace=True)
        #print(driver_encoding_ranked)

        self.data.drop(columns=["ema_past","n_past"],inplace=True) # drop cols used to calculate driver encoding
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def create_relative_driver_race_features(self) -> None:
        relative_race_features = self.data.groupby('Race_Date_Code')[['Sector1Time','Sector2Time','Sector3Time','SpeedST']].transform(lambda x: x / x.max())
        self.data[['Sector1Time_relative','Sector2Time_relative','Sector3Time_relative','SpeedST_relative']] = relative_race_features   
    
    def create_lap_time(self) -> None:
        self.data['lap_time'] = self.data['Sector1Time'] + self.data['Sector2Time'] + self.data['Sector3Time']
    
    def lag_feature(self,feature,alpha=0.6) -> pd.Series:
        # use weighted average of past 2 race data
        data_copy = self.data.copy(deep=True)
        data_copy = data_copy.sort_values(['BroadcastName','Race_Date_Code'])
        return (alpha) * data_copy.groupby('BroadcastName')[feature].shift(1) + (1-alpha) * data_copy.groupby('BroadcastName')[feature].shift(2)
    
    def create_lagged_features(self) -> None:
        self.data['lag_Sector1Time'] = self.lag_feature('Sector1Time')
        self.data['lag_Sector2Time'] = self.lag_feature('Sector2Time')
        self.data['lag_Sector3Time'] = self.lag_feature('Sector3Time')
        self.data['lag_SpeedST'] = self.lag_feature('SpeedST')
        self.data["lag_stint"] = self.lag_feature('Stint',0.5)
        self.data['lag_lap_time'] = self.lag_feature('lap_time')
    
    def create_target(self) -> None:
        self.data["target"] = ((self.data["Race_Position"] <= 3) & (self.data["Race_Position"] > 0)).astype(int)
    
    def create_n_past(self) -> None:
        self.data['n_past']  = self.data.groupby('BroadcastName').cumcount()

    def build_features_into_dataset(self) -> None:
        self.create_top_team()
        self.create_circuit_type()
        self.create_race_date_code()
        self.create_driver_encoding()
        self.create_relative_driver_race_features()
        self.create_lap_time()
        self.create_n_past()
        self.create_lagged_features()
        self.data = self.data.dropna()
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)
        # Encode Round + Year in [0,1]. 1 being the most recent race
        min_code = self.data['Race_Date_Code'].min()
        max_code = self.data['Race_Date_Code'].max()
        self.data['Race_Date_Code'] = (self.data['Race_Date_Code'] - min_code) / (max_code - min_code)
        self.create_target()
    
if __name__ == "__main__":
    dataset = Dataset_v1("data/train_data.csv","data/test_data.csv")
    dataset.build_features_into_dataset()
    data = dataset.get_data()
    print(data.shape)