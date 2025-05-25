import pandas as pd 
from typing import Dict, List

def optimize(betting_odds:Dict[str,float],betting_odds_type:str,total_investment:float,betting_site:str) -> pd.DataFrame:
    preds_df = pd.read_csv("data/preds.csv")

    preds_df['betting_odds'] = preds_df['BroadcastName'].map(betting_odds)
    if betting_odds_type != "decimal":
        #convert to decimal 
        def convert_to_decimal(x):
            if x<0: 
                return 1 + 100/abs(x)
            else: 
                return 1 + x/100

        preds_df['betting_odds'] = preds_df['betting_odds'].apply(convert_to_decimal)

    # convert betting odds to probabilities 
    preds_df['market_prob'] = 1 / preds_df['betting_odds']

    preds_df["is_value_bet"] = preds_df["probs"] > preds_df["market_prob"]

    #calculate expected value of $1 investment. EV = (Pwin * payout) * ((1-Pwin) * Payout) --> simplies to (Pwin * payout) - 1
    preds_df["EV"] = preds_df["probs"] * preds_df["betting_odds"] - 1
    preds_df = preds_df[preds_df["EV"] > 0] # handle negative EVs. we only want to bet where we think we have the edge.
    total_ev = preds_df["EV"].sum()
    print(f"Total EV for {betting_site}: {total_ev}")
    preds_df["stake"] = total_investment * (preds_df["EV"]/total_ev)
    preds_df["stake"] = preds_df["stake"].round(2)

    preds_df.drop(columns=["Qual_Position","driver_encoding"], inplace=True)
    
    preds_df = preds_df[[
        'Year','Round_Number','Location','BroadcastName','preds','probs','market_prob','betting_odds','is_value_bet','EV','stake'
    ]]

    if preds_df["stake"].sum() > total_investment: 
        raise Exception("Total stake exceeds total investment")
    
    preds_df["betting_site"] = betting_site

    preds_df.dropna(inplace=True)
    
    return preds_df

def group_optimize(bet_dfs:List[pd.DataFrame]) -> pd.DataFrame:
    # Concatenate DataFrames row-wise
    combined_df = pd.concat(bet_dfs, ignore_index=True)
    return combined_df


def main():
    TOTAL_INVESTMENT = 10.0

    # Bet MGM
    mgm_odds = {"L NORRIS":1.16,"C LECLERC":1.28,"O PIASTRI":1.44}
    mgm_odds_type = "decimal"
    mgm_betting_site = "Bet MGM"
    mgm_bets = optimize(mgm_odds,mgm_odds_type,TOTAL_INVESTMENT,mgm_betting_site)

    # Draft Kings
    draft_kings_odds = {"L NORRIS":-900,"C LECLERC":-500,"O PIASTRI":-400}
    draft_kings_odds_type = "american"
    draft_kings_betting_site = "Draft Kings"
    draft_kings_bets = optimize(draft_kings_odds,draft_kings_odds_type,TOTAL_INVESTMENT,draft_kings_betting_site)

    # Fanduel
    fanduel_odds = {"L NORRIS":-700,"C LECLERC":-440,"O PIASTRI":-300}
    fanduel_odds_type = "american"
    fanduel_betting_site = "Fanduel"
    fanduel_bets = optimize(fanduel_odds,fanduel_odds_type,TOTAL_INVESTMENT,fanduel_betting_site)

    # Bet365
    bet365_odds = {"L NORRIS":-900,"C LECLERC":-500,"O PIASTRI":-300}
    bet365_odds_type = "american"
    bet365_betting_site = "Bet365"
    bet365_bets = optimize(bet365_odds,bet365_odds_type,TOTAL_INVESTMENT,bet365_betting_site)

    bet_dfs = [mgm_bets,draft_kings_bets,fanduel_bets,bet365_bets]
    grouped_bets = group_optimize(bet_dfs)
    print(grouped_bets)

    grouped_bets_by_driver = grouped_bets.sort_values(by=["probs","BroadcastName"],ascending=[False,True])
    print(grouped_bets_by_driver)

    # Calculate expected profit 
    grouped_bets['expected_profit'] = grouped_bets['EV'] * grouped_bets['stake']
    site_profit = grouped_bets.groupby('betting_site')['expected_profit'].sum()
    site_profit_df = site_profit.reset_index()
    site_profit_df.columns = ['betting_site', 'expected_profit']
    site_profit_df['total_payout'] = TOTAL_INVESTMENT + site_profit_df['expected_profit']
    site_profit_df = site_profit_df.sort_values('expected_profit', ascending=False)
    print(site_profit_df)

    
    
    
if __name__ == "__main__":
    main()
