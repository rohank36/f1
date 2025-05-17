import pandas as pd

def rank_models():
    df = pd.read_csv("model_results.csv")
    df = df.loc[df["data_type"] == "val",["timestamp","dataset_name","features","auroc","recall","precision","f1","f05"]]
    df = df.sort_values(by="f1", ascending=False)
    return df

if __name__ == "__main__":
    df = rank_models()
    print(df)
    
