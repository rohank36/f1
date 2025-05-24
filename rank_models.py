import pandas as pd

def rank_models(set_type:str):
    df = pd.read_csv("model_results.csv")
    df = df.loc[df["data_type"] == set_type,["timestamp","dataset_name","features","model_name","auroc","recall","precision","f1","f05"]]
    df = df.sort_values(by="f1", ascending=False)
    return df

if __name__ == "__main__":
    df = rank_models("val")
    print(df)
    print("\n\n")
    df = rank_models("test")
    print(df)
    
