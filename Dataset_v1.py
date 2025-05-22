import pandas as pd
from Dataset import Dataset
from typing import List, Union
import numpy as np

class Dataset_v1(Dataset):

    def __init__(self,trn_data_path:str,test_data_path:str,verbose=True):
        super().__init__("Dataset_v1",trn_data_path,test_data_path)
        self.verbose = verbose

    def clean_dataset(self) -> None:
        # clean event type
        self.data["Event_Type"] = (self.data["Event_Type"] == "Race").astype(int)
        
        # clean Race NaNs
        cols_to_fill = ["Sprint_Qual_Position", "Sprint_Race_Position","Sector1Time", "Sector2Time", "Sector3Time", "SpeedST", "Stint", "Standardized_Time" ,"Race_Position"]
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
        #if len(self.data["BroadcastName"].unique()) != 31: raise Exception("Wrong number of unique driver names") #uncomment this before pushing, was only to test somehting
        
        self.data = self.data.reset_index(drop=True)

    def train_val_test_split(self, is_for_pred) -> tuple[pd.DataFrame,pd.Series,Union[pd.DataFrame,None],Union[pd.Series,None],pd.DataFrame,Union[pd.Series,None]]:
        data_copy = self.data.copy(deep=True)
        #if "Race_Position" in data_copy.columns: data_copy.drop(columns=["Race_Position"], inplace=True)
        if "target" not in data_copy.columns: raise Exception("Target column not found in data")
        if "target" in self.features_for_training: raise Exception("Target column found in features for training")
        #data_copy = data_copy[self.features_for_training]

        if is_for_pred:
            max_round = data_copy.loc[data_copy["Year"] == 2025, "Round_Number"].max()
            trn_data = data_copy.loc[
                (data_copy["Year"] < 2025) | 
                ((data_copy["Year"] == 2025) & (data_copy["Round_Number"] < max_round)),
                :
            ]

            test_data = data_copy.loc[
                (data_copy["Year"] == 2025) & (data_copy["Round_Number"] >= max_round),
                :
            ]
            y_trn = trn_data["target"]
           
            x_trn = trn_data[self.features_for_training]
            x_test = test_data[self.features_for_training]

            print("\nDataset Shapes")
            print(x_trn.shape,y_trn.shape)
            print(x_test.shape)
            print("\n")

            return x_trn,y_trn,None,None,x_test,None
        
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

        print("\nDataset Shapes")
        print(x_trn.shape,y_trn.shape)
        print(x_val.shape,y_val.shape)
        print(x_test.shape, y_test.shape)
        print("\n")
             
        return x_trn,y_trn,x_val,y_val,x_test,y_test

    def set_features_for_training(self,selected_features:List[str]) -> None:
        self.features_for_training:List[str] = selected_features
        print("\nFeatures for training set:\n",self.features_for_training)

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
    
    def create_last_race_position(self) -> None:
        self.data['last_race_position'] = self.data.groupby('BroadcastName')['Race_Position'].shift(1)
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

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

    def create_pos_gained_encoding(self) -> None:
        self.data = self.data.sort_values(['BroadcastName', 'Race_Date_Code'])

        qual_position = self.data['Qual_Position']
        race_position = self.data['Race_Position']

        # Handle -1 in race_position (assume DNF or did not start)
        race_position = race_position.where(race_position != -1, 21)

        pos_gained = qual_position - race_position

        # Finishing weight calculation
        finishing_weight = (1 / np.log(race_position + 1)) * ((21 - race_position) / 20)

        # Precompute log term safely
        safe_log_term = -np.log(finishing_weight.clip(lower=1e-6))

        # Vectorized final score calculation
        adjusted_pos_gained = (
            (pos_gained == 0) * finishing_weight + 
            (pos_gained > 0) * (pos_gained * finishing_weight) + 
            (pos_gained < 0) * (pos_gained * safe_log_term * 0.05)
        )

        # Add podium bonus
        adjusted_pos_gained += (race_position <= 3) * 1  # podium_bonus = 1

        # Store final scores directly
        self.data['final_score'] = adjusted_pos_gained

        # Exponential Weighted Average Encoding
        self.data['n_past_final'] = self.data.groupby('BroadcastName').cumcount()
        global_mean_final = self.data['final_score'].mean()
        k = 20
        alpha = 0.3

        self.data['ema_final'] = (
            self.data
            .groupby('BroadcastName', group_keys=False)['final_score']
            .apply(lambda x: x.shift(1).ewm(alpha=alpha, adjust=True).mean())
        )

        self.data['pos_gained_encoding'] = (
            (self.data['n_past_final'] * self.data['ema_final'] + k * global_mean_final) /
            (self.data['n_past_final'] + k)
        )

        # Validate feature
        self.check_feature("pos_gained_encoding")

        # Clean up
        self.data.drop(columns=["ema_final", "n_past_final", "final_score"], inplace=True)
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def create_pos_gained_encoding_simple(self) -> None:
        self.data = self.data.sort_values(['BroadcastName', 'Race_Date_Code'])

        qual_position = self.data['Qual_Position']
        race_position = self.data['Race_Position']

        # Handle -1 in race_position (assume DNF or did not start)
        race_position = race_position.where(race_position != -1, 21)

        pos_gained = qual_position - race_position

        # Finishing weight calculation
        finishing_weight = (1 / np.log(race_position + 1)) * ((21 - race_position) / 20)

        # Precompute log term safely
        safe_log_term = -np.log(finishing_weight.clip(lower=1e-6))

        # Vectorized final score calculation
        adjusted_pos_gained = (
            (pos_gained == 0) * finishing_weight + 
            (pos_gained > 0) * (pos_gained * finishing_weight) + 
            (pos_gained < 0) * (pos_gained * safe_log_term * 0.05)
        )

        # Add podium bonus
        adjusted_pos_gained += (race_position <= 3) * 1  # podium_bonus = 1

        # Store final scores directly
        self.data['final_score'] = adjusted_pos_gained

        # Create pos_gained_encoding based only on the last race result
        self.data['pos_gained_encoding_simple'] = (
            self.data
            .groupby('BroadcastName', group_keys=False)['final_score']
            .shift(1)  # Previous race's final_score
        )

        # Validate feature
        self.check_feature("pos_gained_encoding_simple")

        # Clean up
        self.data.drop(columns=["final_score"], inplace=True)
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def create_qual_q3_time(self) -> None:
        # !!NOTE: SOME OF THE TIMES IN THIS FEATURE SEEM WRONG (THIS IS AN ERROR FROM THE DATA LOADING PORTION PEICE, NOT HERE - JUST USE THIS FEATURE WITH CAUTION )
        self.data = self.data.sort_values(['Race_Date_Code'])
        def normalize_q3_times(group):
            # Keep -1 values as is, normalize only real times
            real_times = group[group != -1]
            if real_times.empty:
                return group  # No real times to normalize

            min_time = real_times.min()
            return group.apply(lambda x: x - min_time if x != -1 else -1)

        self.data["Qual_Q3_Time_Normal"] = (
            self.data
            .groupby("Race_Date_Code", group_keys=False)["Qual_Q3_Time"]
            .transform(normalize_q3_times)
        )

        self.data.drop(columns=["Qual_Q3_Time"],inplace=True) # drop cols used to calculate driver encoding
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)
    
    def create_race_time_encoding_v1(self) -> None:
        """
        Note: a lot of race_times for this feature from the api were wrong, but the top 5 driver times were all correct that is why you're masking it and only using the top 5.
        """
        def mask_group(group):
            # Sort by Race_Position ascending
            group = group.sort_values("Race_Position", ascending=True).copy()
            
            # Create a mask: keep top 10, NaN the rest
            group["Standardized_Time"] = group["Standardized_Time"].where(group["Race_Position"] <= 5, np.nan)
        
            return group

        def check_standardized_time_ordering(df):
            def is_time_ascending(group):
                # Sort by Race_Position
                group_sorted = group.sort_values("Race_Position", ascending=True)
                # Check if Standardized_Time is non-decreasing
                valid_times = group_sorted["Standardized_Time"].dropna()
                return valid_times.is_monotonic_increasing

            # Apply the check to each (Year, Round_Number) group
            result = (
                df.groupby(["Year", "Round_Number"])
                .apply(is_time_ascending)
                .reset_index(name="is_time_ordered")
            )

            return result

        # Apply to each (Year, Round_Number) group
        self.data = self.data.groupby(["Year", "Round_Number"], group_keys=False).apply(mask_group)

        res = check_standardized_time_ordering(self.data)
        if len(res.loc[res["is_time_ordered"]==False]) > 0: 
            print(res.loc[res["is_time_ordered"]==False])
            raise Exception("Standardized_Time is not ordered for some races")
        
        self.data = self.data.sort_values(['BroadcastName','Race_Date_Code']) 

        self.data['Race_Time_Encoding'] = (
            self.data
            .groupby('BroadcastName', group_keys=False)['Standardized_Time']
            .shift(1)  # Previous race's final_score
        )
       
        self.data["Race_Time_Encoding"].replace(np.nan, 1000, inplace=True)
        self.data["Race_Time_Encoding"].replace(-1, 1000, inplace=True)
        self.check_feature("Race_Time_Encoding")
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def create_race_time_encoding(self) -> None:
        """
        Note: a lot of race_times for this feature from the api were wrong, but the top 5 driver times were all correct that is why you're masking it and only using the top 5.
        """
        def mask_group(group):
            # Sort by Race_Position ascending
            group = group.sort_values("Race_Position", ascending=True).copy()
            
            # Create a mask: keep top 10, NaN the rest
            group["Standardized_Time"] = group["Standardized_Time"].where(group["Race_Position"] <= 3, np.nan)
        
            return group

        def check_standardized_time_ordering(df):
            def is_time_ascending(group):
                # Sort by Race_Position
                group_sorted = group.sort_values("Race_Position", ascending=True)
                # Check if Standardized_Time is non-decreasing
                valid_times = group_sorted["Standardized_Time"].dropna()
                return valid_times.is_monotonic_increasing

            # Apply the check to each (Year, Round_Number) group
            result = (
                df.groupby(["Year", "Round_Number"])
                .apply(is_time_ascending)
                .reset_index(name="is_time_ordered")
            )

            return result

        # Apply to each (Year, Round_Number) group
        self.data["Standardized_Time"].replace(-1, np.nan, inplace=True) #dealing with edge cases
        self.data = self.data.groupby(["Year", "Round_Number"], group_keys=False).apply(mask_group)

        res = check_standardized_time_ordering(self.data)
        if len(res.loc[res["is_time_ordered"]==False]) > 0: 
            print(res.loc[res["is_time_ordered"]==False])
            raise Exception("Standardized_Time is not ordered for some races")
        
        self.data = self.data.sort_values(['BroadcastName','Race_Date_Code']) 

        def normalize_standardized_time(group):
            times = group['Standardized_Time']
            return (times - times.min()) / (times.max() - times.min() + 1e-6)

        self.data['Standardized_Time_Normalized'] = (
            self.data
            .groupby(['Year', 'Round_Number'], group_keys=False)
            .apply(lambda g: normalize_standardized_time(g))
            .fillna(1.0)
        )

        self.data['Race_Time_Encoding'] = (
            self.data
            .groupby('BroadcastName', group_keys=False)['Standardized_Time_Normalized']
            #.shift(1)  # Previous race's final_score
            .apply(lambda x: x.shift(1).ewm(alpha=0.4, adjust=True).mean())
            #.fillna(1.0)
        )
       
        #self.data["Race_Time_Encoding"].replace(np.nan, 1000, inplace=True)
        #self.data["Race_Time_Encoding"].replace(-1, 1000, inplace=True)
        self.data.drop(columns=["Standardized_Time_Normalized"], inplace=True)
        self.check_feature("Race_Time_Encoding")
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def create_n_past_race_wins(self) -> None:
        """Create a feature that counts the number of past race wins for each driver"""
        # Sort by driver and race date
        self.data = self.data.sort_values(['BroadcastName','Race_Date_Code'])
        
        # Create a binary win indicator (1 if Race_Position == 1, 0 otherwise)
        self.data['race_win'] = (self.data['Race_Position'] == 1).astype(int)
        
        # Calculate cumulative sum of wins, shifted by 1 to exclude current race
        self.data['n_past_race_wins'] = (
            self.data
            .groupby('BroadcastName')['race_win']
            .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
            .fillna(0)
        )
        
        # Drop the temporary race_win column
        self.data = self.data.drop('race_win', axis=1)
        
        self.check_feature("n_past_race_wins")

        # Sort back to chronological order
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)

    def build_features_into_dataset(self) -> None:
        self.create_target()

        self.create_top_team()
        self.create_circuit_type()
        self.create_race_date_code()
        self.create_driver_encoding()
        self.create_race_time_encoding()
        self.create_qual_q3_time()
        self.create_pos_gained_encoding()
        self.create_pos_gained_encoding_simple()
        self.create_ewa_driver_results() #this creates NaN for the 1st race
        #self.create_relative_driver_race_features()
        self.create_n_past_podiums()
        self.create_n_past_podiums_last_5()
        self.create_n_past_race_wins()
        self.create_lap_time()
        self.create_n_past()
        #self.create_last_race_position() # retire this feature. Just adds noise due to high correlation with driver encoding.
        self.create_lagged_features() # this creates NaN for the 1st and 2nd race 

        columns_to_drop = ["Sector1Time","Sector2Time","Sector3Time","SpeedST","Stint","lap_time","Standardized_Time"] #drop race data
        self.data = self.data.drop(columns=columns_to_drop)

        self.data = self.data.dropna() # you lose the first 2 of each driver because of this
        self.data = self.data.sort_values(['Race_Date_Code']).reset_index(drop=True)
        # Encode Round + Year in [0,1]. 1 being the most recent race
        #min_code = self.data['Race_Date_Code'].min()
        #max_code = self.data['Race_Date_Code'].max()
        #self.data['Race_Date_Code'] = (self.data['Race_Date_Code'] - min_code) / (max_code - min_code)
        
    
if __name__ == "__main__":
    dataset = Dataset_v1("data/train_data.csv","data/test_data.csv")
    dataset.build_features_into_dataset()
    data = dataset.get_data()
    print(data.shape)