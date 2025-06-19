# 🏎️ F1 Podium Prediction Model
A machine learning project that predicts whether an F1 driver will podium or not. 

## 🧠 Goal
Use model predictions to make bets on online sports gambling platforms. To date the model has achieved a 35% return.


## 📊 Features Used

- Qualifying position  
- Driver Performance Encoding (Engineered feature) 


## 📈 Results

| Metric     | Score |
| ---------- | ----- |
| Accuracy   | 0.95  |
| AUROC      | 0.98  |
| Precision  | 0.80  |
| Recall     | 0.95  |
| F1-Score   | 0.87  |
| Top-3 Acc. | 0.88  |

## 🧪 Future Improvements
- Look at alternative data sources to find predictive alpha (e.g. twitter/X, news stories etc)
- Experiment with Free Practice times
- Try other ensemble methods (XGBoost, LightGBM)
- Auto-update data and run new race prediction cron jobs