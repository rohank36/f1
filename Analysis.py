import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from Dataset import Dataset
from typing import List, Dict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso

class FeatureAnalysis:
    def __init__(self,dataset:Dataset):
        self.data:pd.DataFrame = dataset.get_data()
        self.non_numeric_features = self.get_non_numeric_features()
        print(f"Non numeric features:\n {self.non_numeric_features}")
        self.data.drop(columns=self.non_numeric_features, inplace=True)
        self.race_position = self.data["Race_Position"]
        self.data.drop(columns=["Race_Position"], inplace=True)
        if "target" not in self.data.columns: raise Exception("Target column not found in data")

    def analyze_auroc_score(self) -> pd.DataFrame:
        # here the higher or lower the auc the better it separates the two classes. Things closer to 0.5 are not good.
        feature_auc_df = pd.DataFrame()
        for feature in self.data.columns:
            if feature == "target": continue
            auc = roc_auc_score(self.data["target"],self.data[feature])           
            feature_auc_df = pd.concat([feature_auc_df,pd.DataFrame([{'feature':feature,'auc':auc}])],axis=0,ignore_index=True)
        return feature_auc_df.sort_values(by='auc',ascending=False)

    def plot_prob_dist(self, feature: str) -> None:
        plt.figure(figsize=(20, 10))
        
        # Calculate statistics
        feature_data = self.data[feature]
        skewness = feature_data.skew()
        kurtosis = feature_data.kurt()
        mean = feature_data.mean()
        std = feature_data.std()
        
        # Plot histogram
        plt.hist(feature_data, alpha=0.5, bins=30, edgecolor='black')
        
        # Add statistics to plot
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(f"{feature} Distribution\nSkewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}")
        
        # Add vertical lines for mean and mean ± std
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(mean + std, color='orange', linestyle='--', label=f'Mean + 1σ: {mean + std:.2f}')
        plt.axvline(mean - std, color='orange', linestyle='--', label=f'Mean - 1σ: {mean - std:.2f}')
        
        # Add legend
        plt.legend()
        
        plt.show()

    def plot_feature_vs_target(self,feature:str) -> None:
        plt.figure(figsize=(20,10))
        plt.scatter(self.data[feature],self.race_position,alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel("Target")
        plt.title(f"{feature} vs Target")
        plt.show()

    def lasso_feature_weights(self, alpha: float = 1.0) -> pd.DataFrame:
        # Prepare data
        X = self.data.drop(columns=['target'])
        y = self.data['target']
        
        # Initialize and fit Lasso model
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X, y)
        
        # Create DataFrame of feature weights
        weights_df = pd.DataFrame({
            'feature': X.columns,
            'weight': lasso.coef_
        })
        
        # Sort by absolute value of weights (most important features first)
        weights_df['abs_weight'] = weights_df['weight'].abs()
        weights_df = weights_df.sort_values('abs_weight', ascending=False)
        
        return weights_df

    def analyze_feature_separation(self,feature: str, print_plt: bool = True, print_stats: bool = True) -> Dict[str, float]:
        if print_plt:
            plt.figure(figsize=(20,10))
            
            target = self.data["target"]

            plt.scatter(self.data.index[target == 0], self.data.loc[target == 0, feature], 
                        color='blue', alpha=0.3, label='Not Podium', 
                        edgecolors='none')
            plt.scatter(self.data.index[target == 1], self.data.loc[target == 1, feature], 
                        color='red', alpha=0.8, label='Podium', 
                        edgecolors='none')
            
            plt.title(f'{feature} Over Time Colored by Race Position')
            plt.xlabel('Data Point Index')
            plt.ylabel(feature)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        # Separate the classes
        podium = self.data[self.data["target"] == 1][feature]
        non_podium = self.data[self.data["target"] == 0][feature]
    
        # T-test to check if means are significantly different
        t_statistic, p_value = stats.ttest_ind(podium, non_podium)
    
        # Cohen's d for effect size
        cohens_d = (podium.mean() - non_podium.mean()) / np.sqrt((podium.std()**2 + non_podium.std()**2) / 2)

        if print_stats:
            # Statistical measures
            print(f"Podium class statistics for {feature}:")
            print(f"Mean: {podium.mean()}")
            print(f"Median: {podium.median()}")
            print(f"Standard Deviation: {podium.std()}")
        
            print(f"\nNon-podium class statistics for {feature}:")
            print(f"Mean: {non_podium.mean()}")
            print(f"Median: {non_podium.median()}")
            print(f"Standard Deviation: {non_podium.std()}")

            print(f"\nt-test results:")
            print(f"t-statistic: {t_statistic} --> difference between group means relative to variation in data. Larger = greater separation.")
            print(f"p-value: {p_value} --> probability of observing the differnce by randomn chance. Smaller = better")
            

            print(f"Cohen's d (effect size): {cohens_d} --> Higher abs value = difference is big and meaningful")
        
        return {
            'podium_mean': podium.mean(),
            'non_podium_mean': non_podium.mean(),
            't_statistic': t_statistic,
            'p_value': p_value,
            'cohens_d': cohens_d
        }

    def get_feature_separation_ranking(self,features_to_analyze:List[str]) -> pd.DataFrame:
        # Dictionary to store separation results
        separation_results = {}
        
        # Analyze each feature
        for feature in features_to_analyze:
            try:
                if feature == "target":
                    continue
                
                result = self.analyze_feature_separation(feature, False, False)
                
                # Create a separation score 
                # Combine absolute Cohen's d with statistical significance
                # Higher absolute Cohen's d and lower p-value = better separation
                separation_score = abs(result['cohens_d']) / (result['p_value'] + 1e-10)
                
                separation_results[feature] = {
                    'separation_score': separation_score,
                    'cohens_d': result['cohens_d'],
                    'p_value': result['p_value'],
                    'podium_mean': result['podium_mean'],
                    'non_podium_mean': result['non_podium_mean']
                }
            except Exception as e:
                print(f"Could not analyze {feature}: {e}")
        
        # Convert to DataFrame and sort
        separation_df = pd.DataFrame.from_dict(separation_results, orient='index')
        separation_df.index.name = 'feature'
        separation_df = separation_df.reset_index()
        
        # Sort by separation score in descending order
        return separation_df.sort_values('separation_score', ascending=False)

    def corr_heat_map(self) -> None:
        plt.figure(figsize=(10,8))
        sns.heatmap(self.data.corr().loc[:,["target"]], 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0)
        plt.title('Correlation with Target')
        plt.tight_layout()
        plt.show()

    def get_precision_matrix(self) -> pd.DataFrame:
        precision_matrix_df = pd.DataFrame(
            np.linalg.inv(self.data.cov()), 
            columns=self.data.columns, 
            index=self.data.columns
        )
        precision_matrix_df = precision_matrix_df.round(4)
        return precision_matrix_df.loc[:,["target"]]

    def select_features(self, target='target') -> pd.DataFrame:
        # Combine multiple methods to see predictive power of feature for Race_Position

        # 1. Correlation Analysis
        correlation = self.data.corr()[target].abs().sort_values(ascending=False)
        
        # 2. Precision Matrix Analysis
        precision_matrix = pd.DataFrame(
            np.linalg.inv(self.data.cov()), 
            columns=self.data.columns, 
            index=self.data.columns
        )
        precision_with_target = precision_matrix.loc[:, target].abs()
        
        # 3. Feature Separation Ranking (from previous function)
        separation_ranking = self.get_feature_separation_ranking(self.data.columns)
        
        # 4. AUROC Scores
        auroc_scores = self.analyze_auroc_score()
        
        # 5. Lasso Regression Weights
        lasso_weights = self.lasso_feature_weights()
        
        # Combine results
        feature_scores = pd.DataFrame({
            'Correlation': correlation,
            'Precision': precision_with_target,
            'Separation_Score': separation_ranking.set_index('feature')['separation_score'],
            'AUROC': auroc_scores.set_index('feature')['auc'],
            'Lasso_Weight': lasso_weights.set_index('feature')['weight']
        }).sort_values('Correlation', ascending=False)
        feature_scores.drop(target, inplace=True)
        return feature_scores

    def get_non_numeric_features(self) -> List[str]:
        non_numeric_features = []
        for feature in self.data.columns:
            if not np.issubdtype(self.data[feature].dtype, np.number):
                non_numeric_features.append(feature)
        return non_numeric_features