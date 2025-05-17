import pandas as pd
from Dataset import Dataset
from typing import List

class Dataset_v1(Dataset):

    def __init__(self,trn_data_path:str,test_data_path:str,verbose=True):
        super().__init__("Dataset_v1",trn_data_path,test_data_path)
        self.verbose = verbose

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

    def train_val_test_split(self) -> tuple[pd.DataFrame,pd.Series,pd.DataFrame,pd.Series,pd.DataFrame,pd.Series]:
        data_copy = self.data.copy(deep=True)
        #if "Race_Position" in data_copy.columns: data_copy.drop(columns=["Race_Position"], inplace=True)
        if "target" not in data_copy.columns: raise Exception("Target column not found in data")
        if "target" in self.features_for_training: raise Exception("Target column found in features for training")
        #data_copy = data_copy[self.features_for_training]
        
        trn_data = data_copy.loc[data_copy["Year"]!=2025,:]
        test_data = data_copy.loc[data_copy["Year"]==2025,:]

        val_mask = (data_copy["Year"] == 2024) & (data_copy["Round_Number"] > 12)
        val_data = data_copy[val_mask]
        trn_data = data_copy[~val_mask]

        y_trn = trn_data["target"]
        y_val = val_data["target"]
        y_test = test_data["target"]

        x_trn = trn_data[self.features_for_training]
        x_val = val_data[self.features_for_training]
        x_test = test_data[self.features_for_training]

        print(x_trn.shape,y_trn.shape)
        print(x_val.shape,y_val.shape)
        print(x_test.shape, y_test.shape)
             
        return x_trn,y_trn,x_val,y_val,x_test,y_test

    def set_features_for_training(self,selected_features:List[str]) -> None:
        self.features_for_training:List[str] = selected_features
        print("Features for training set:\n",self.features_for_training)

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

        self.check_feature("driver_encoding")

        self.data.drop(columns=["ema_past","n_past"],inplace=True) # drop cols used to calculate driver encoding
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def create_ewa_driver_results(self) -> None:
        # Raw recent results, heavily weighted. Doesn't care about driver experience. 
        # Sort by driver and race date
        sorted_data = self.data.sort_values(['BroadcastName','Race_Date_Code'])
        
        # Calculate EWMA for each driver
        ewa_results = sorted_data.groupby('BroadcastName')['Race_Position'].transform(
            lambda x: x.shift(1).ewm(alpha=0.5, adjust=True).mean()
        )
        
        # Create new column with aligned index
        self.data['ewa_driver_results'] = ewa_results

        self.check_feature("ewa_driver_results")
        
        # Sort back to chronological order
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

    def check_feature(self,feature) -> None:
        if not self.verbose: return
        df = (
            self.data
            .sort_values(['BroadcastName','Race_Date_Code'])
            .groupby('BroadcastName')[feature]
            .last()
            .reset_index()
        )
        df.sort_values(by=feature, inplace=True)
        print(df)

    def create_n_past_podiums(self) -> None:
        self.data = self.data.sort_values(['BroadcastName','Race_Date_Code'])
        self.data['n_past_podiums'] = (
            self.data
            .groupby('BroadcastName')['target']
            .cumsum()
            .shift(1)
            .fillna(0)
        )
        self.check_feature("n_past_podiums")

        # Sort back to chronological order
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def create_n_past_podiums_last_5(self) -> None:
        # Ensure data is sorted properly for rolling operations
        self.data = self.data.sort_values(['BroadcastName', 'Race_Date_Code'])

        self.data['n_past_podiums_last_5'] = (
            self.data
            .groupby('BroadcastName')['target']
            .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
            .fillna(0)
        )

        self.check_feature("n_past_podiums_last_5")

        # Sort back to chronological order
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def build_features_into_dataset(self) -> None:
        self.create_target()

        self.create_top_team()
        self.create_circuit_type()
        self.create_race_date_code()
        self.create_driver_encoding()
        self.create_ewa_driver_results() #this creates NaN for the 1st race
        #self.create_relative_driver_race_features()
        self.create_n_past_podiums()
        self.create_n_past_podiums_last_5()
        self.create_lap_time()
        self.create_n_past()
        self.create_lagged_features() # this creates NaN for the 1st and 2nd race 

        self.data = self.data.dropna() # you lose the first 2 of each driver because of this
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)
        # Encode Round + Year in [0,1]. 1 being the most recent race
        #min_code = self.data['Race_Date_Code'].min()
        #max_code = self.data['Race_Date_Code'].max()
        #self.data['Race_Date_Code'] = (self.data['Race_Date_Code'] - min_code) / (max_code - min_code)
        columns_to_drop = ["Sector1Time","Sector2Time","Sector3Time","SpeedST","Stint","lap_time"] #drop race data
        self.data = self.data.drop(columns=columns_to_drop)
    
if __name__ == "__main__":
    dataset = Dataset_v1("data/train_data.csv","data/test_data.csv")
    dataset.build_features_into_dataset()
    data = dataset.get_data()
    print(data.shape)