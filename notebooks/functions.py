import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_score
import optuna
import pickle

def load_data(path_to_file):
    
    """
    Load a csv file into a dataframe and display a brief overview of the first 5 rows.
    """
    
    df = pd.read_csv(path_to_file)

    return df

def clean_datatype(df, float_columns = None, int_columns = None, str_columns = None, date_columns = None, bool_columns = None):
    
    """
    Converts specified columns in the dataframe to the desired data types. 
    
    Requirement: 
    - No missing values
    
    Parameters:
    - float_columns: List of column names to convert to float
    - int_columns: List of column names to convert to int
    - str_columns: List of column names to convert to string
    - date_columns: List of column names to convert to datetime
    - bool_columns: List of column names containing boolean-like strings ('t', 'true', 'f', 'false') to convert to int
    
    Returns:
    - Dataframe with updated column types
    """

    # Default empty lists if no list is provided
    int_columns = int_columns or []
    str_columns = str_columns or []
    date_columns = date_columns or []
    bool_columns = bool_columns or []
    
    if float_columns:
        for col in float_columns:
            df[col] = df[col].astype(float)

    if int_columns:
        for col in int_columns:
            df[col] = df[col].astype(int)
    
    if str_columns:
        for col in str_columns:
            df[col] = df[col].astype(str)
    
    if date_columns:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col]) 
            # Adjust the datetime conversion here if the format looks different
    
    if bool_columns:
        # Convert true and false to integer
        for col in bool_columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['t', 'true'] else 0)

    return df

def clean_channel_sales(df):

    """
    Converts unconvenient sales channel names to integer dummies with respect to their frequency
    """

    # Remove rows belonging to channels with value counts below 20 can be removed because they don't have much impact on the final model
    valid_channels = df.channel_sales.value_counts()[df.channel_sales.value_counts() > 20].index
    df = df[df.channel_sales.isin(valid_channels)]

    # Rename remaining channels for simplicity according to their rank in frequency
    frequency_map = df.channel_sales.value_counts().rank(method='dense', ascending=False).astype(int) - 1
    df.loc[:, 'channel_sales'] = df.channel_sales.map(frequency_map)
                                                            
    return df

def save_to_pickle(obj, file_path):
    """
    Save an object (scaler or model) in a pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_from_pickle(file_path):
    """
    Load an object (scaler or model) in a pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def plot_categorical(df, categorical_column):
    
    """
    Plots the churn and no churn distribution of a categorical column in a stacked bar plot normalized to 100%.
    
    Parameters:
    - df: DataFrame containing the data
    - categorical_column: The categorical column to plot
    """
    churn_color = '#10b38f'
    no_churn_color = '#84aaff'
    
    # Creates a dataframe with three columns: the categorical_column, churn, and the corresponding count of occurrences.
    churn_counts = df.groupby([categorical_column, 'churn']).size().reset_index(name = 'count')

    # Calculates the percentage of each churn category within each group of the categorical_column
    churn_counts['percentage'] = churn_counts.groupby(categorical_column)['count'].transform(lambda x: x / x.sum() * 100)

    # Maps the numeric values in the churn column (e.g., 1 and 0) to descriptive labels ('churn' and 'no churn').
    churn_counts['churn'] = churn_counts['churn'].map({1: 'churn', 0: 'no churn'})
    
    fig = px.bar(churn_counts, 
                 x = categorical_column, 
                 y = 'percentage', 
                 color = 'churn',
                 color_discrete_map = {'churn': churn_color, 'no churn': no_churn_color})

    fig.update_layout(xaxis_title = categorical_column, 
                      yaxis_title = "Percentage (%)",
                      title = 'Histogram',
                      bargap = 0.2,
                      barmode = 'stack',
                      margin = dict(t = 40, b = 30),
                      height = 400,
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    
    return fig

def plot_numerical(df, numerical_column, bins):
    
    """
    Plots the churn and no churn distribution of a numerical column in a stacked histogram normalized to 100%.

    Parameters:
    - df: DataFrame containing the data
    - numerical_column: The numerical column to plot
    - bins: Number of bins for the histograms
    """

    churn_color = '#10b38f'
    no_churn_color = '#84aaff'
    
    # Filter data by churn and no churn
    churn_data = df[df['churn'] == 1][numerical_column].dropna()
    no_churn_data = df[df['churn'] == 0][numerical_column].dropna()
    
    # Compute histograms
    hist_churn, bin_edges = np.histogram(churn_data, bins = bins, density = False)
    hist_no_churn, _ = np.histogram(no_churn_data, bins = bin_edges, density = False)
    
    # Normalize to percentage
    total_counts = hist_churn + hist_no_churn
    total_counts[total_counts == 0] = 1 # prevent division by zero
    hist_churn_normalized = (hist_churn / total_counts) * 100
    hist_no_churn_normalized = (hist_no_churn / total_counts) * 100

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(x = bin_centers, y = hist_no_churn_normalized, name = "no churn", marker_color = no_churn_color))
    fig.add_trace(go.Bar(x = bin_centers, y = hist_churn_normalized, name = "churn", marker_color = churn_color))
    
    fig.update_layout(xaxis_title = numerical_column, 
                    yaxis_title = "Percentage (%)",
                    title = 'Histogram',
                    barmode='stack', 
                    bargap = 0.2,
                    margin = dict(t = 40, b = 30),
                    height = 400,
                    legend_title="churn",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)')
    
    return fig

def plot_kde(df, numerical_column, max, min = 0):
    
    """
    Plots overlaid KDE functions for churn and no churn distributions of a numerical column.

    Parameters:
    - df: DataFrame containing the data
    - numerical_column: The numerical column to plot
    - max: Maximum value shown on x axis
    """

    churn_color = '#10b38f'
    no_churn_color = '#84aaff'

    # Filter data by churn and no churn
    churn_data = df[df['churn'] == 1][numerical_column].dropna()
    no_churn_data = df[df['churn'] == 0][numerical_column].dropna()
    
    # Compute KDE for churn
    churn_kde = gaussian_kde(churn_data)
    churn_x = np.linspace(min, max, 1000)
    churn_y = churn_kde(churn_x)
    
    # Compute KDE for no churn
    no_churn_kde = gaussian_kde(no_churn_data)
    no_churn_x = np.linspace(min, max, 1000)
    no_churn_y = no_churn_kde(no_churn_x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = churn_x, y = churn_y, mode = 'lines', name = 'churn', line=dict(color=churn_color, width=2)))
    fig.add_trace(go.Scatter(x = no_churn_x, y = no_churn_y, mode = 'lines', name = 'no churn', line = dict(color=no_churn_color, width=2)))
    
    fig.update_layout(xaxis_title = numerical_column, 
                      yaxis_title = "Density", 
                      title = 'KDE Plot', 
                      height = 400, 
                      margin = dict(t = 40, b = 30),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    
    return fig


def dummies(df, dummy_columns):
    
    """
    Creates dummies for categorical columns in the dataframe
    
    Parameters:
    - df: DataFrame containing the data
    - dummy_columns: List of categorical columns to create dummies for

    Return:
    - The modified dataframe.
    """

    df = df.copy()

    for item in dummy_columns:

        # Store dummies as integers
        dummies = pd.get_dummies(df[item], prefix = item).astype(int)

        # Concats the original dataframe with the dummies
        df = pd.concat([df, dummies], axis=1)

        # Drops the original column
        df.drop(item, axis=1, inplace=True)
    
    return df

def prepare_data(df, drop_columns, target_column, test_size = 0.2, random_state = 0):
    
    """
    Splits the dataset into train and test sets and scales the features with the Standard Scaler.
    
    Parameters:
    - df: The DataFrame containing features and target.
    - drop_columns: List of columns to drop (e.g., id columns).
    - target_column
    - test_size: Proportion of data to use as test set.
    - random_state: Random seed for reproducibility.
    
    Returns:
    - X_train, X_test, y_train, y_test: Scaled train-test split data.
    - scaler: The fitted StandardScaler object.
    """

    # Define features of the model by dropping any id related columns and the target column
    features = df.drop(columns = drop_columns + [target_column])

    # Define the target column
    target = df[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = test_size, random_state = random_state)

    # Create an instance of the standard scaler
    scaler = StandardScaler()
        
    # Fit the scaler to the train data
    scaler.fit(X_train)

    # Transform the train and test data with the scaler instance
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (pd.DataFrame(X_train_scaled, columns = X_train.columns), 
            pd.DataFrame(X_test_scaled, columns = X_test.columns),
            y_train, y_test, scaler)

def train_random_forest(X_train, y_train, n_estimators = 100, max_depth = 20, min_samples_split = 2, min_samples_leaf = 1):
    
    """
    Trains a Random Forest classifier.
    
    Parameters:
    - X_train: Training features.
    - y_train: Training target.
    - n_estimators: Number of trees in the forest.
    - max_depth: Maximum depth of the trees.
    
    Returns:
    - Trained RandomForestClassifier model.
    """

    # Creates an instance of the random forest classifier with balanced class weights
    model = RandomForestClassifier(n_estimators = n_estimators, 
                                   max_depth = max_depth,
                                   class_weight = 'balanced', 
                                   min_samples_split = min_samples_split,
                                   min_samples_leaf = min_samples_leaf,
                                   random_state = 0)

    # Trains the classifier with the train data
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    
    """
    Evaluates the model on the test set.
    
    Parameters:
    - model: Trained model.
    - X_test: Test features.
    - y_test: Test target.
    
    Returns:
    - A dictionary containing accuracy, precision, and recall scores.
    """

    y_pred = model.predict(X_test)
    metrics = ["Recall", "Precision", "Accuracy"]
    scores = [round(recall_score(y_test, y_pred, average="binary"),2),
                round(precision_score(y_test, y_pred, average="binary"),2),
                round(accuracy_score(y_test, y_pred),2)]

    scores_df = pd.DataFrame({"Metric": metrics, "Value": scores})

    fig = px.bar(scores_df, y = 'Metric', x = 'Value', color = metrics, orientation='h', text = 'Value', title = 'Performance Metrics', height = 200, width = 500)
    
    fig.update_traces(marker_color='#84aaff', texttemplate='%{x:.0%}',)
    
    fig.update_layout(showlegend=False, 
                      margin = dict(t = 40, b = 30), 
                      xaxis_tickformat='.0%', 
                      xaxis_title='Percentage',
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')

    return fig

def train_random_forest_tuned(X_train, y_train, n_trials = 50, cv = 5):
    
    """
    Tunes hyperparameters and trains the RandomForest model with the best configuration.

    Parameters:
    - X_train: Scaled training features.
    - y_train: Training target.
    - n_trials: Number of trials for hyperparameter tuning (default: 50).
    - cv: Number of cross-validation folds for evaluation (default: 5).

    Returns:
    - tuned_model: The trained RandomForest model with the best hyperparameters.
    """
    
    # 50 trials balance exploration and exploitation of the hyperparameter space.

    # 5 folds provide a reliable performance estimate and a reasonable computational cost.
    # More folds would leed to lower biases but increase computational cost quickly, less folds would lead to higher variance.

    def objective(trial):

        """
        Defines the objective function for hyperparameter optimization using Optuna.

        Parameters:
        - trial: An Optuna trial object used to suggest hyperparameter values.

        Returns:
        - The average recall score for the given hyperparameter combination.
        """
    
        # Suggests values for key hyperparameters of the Random Forest model.
        n_estimators = trial.suggest_int('n_estimators', 90, 110)
        max_depth = trial.suggest_int('max_depth', 10, 12)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)

        # Evaluates the model using 5-fold cross-validation with recall as the scoring metric.
        model = RandomForestClassifier(n_estimators = n_estimators, 
                                        max_depth = max_depth,
                                        min_samples_split = min_samples_split, 
                                        min_samples_leaf = min_samples_leaf,
                                        class_weight = 'balanced',
                                        random_state = 0)

        # Define recall scorer to treat the low recall
        recall_scorer = make_scorer(recall_score)

        # Evaluate using cross validation and parallel computation (n_jobs = -1)
        # Finally compute the mean of all recall scores
        score = cross_val_score(model, X_train, y_train, cv = cv, scoring = recall_scorer, n_jobs = - 1).mean()

        return score

    # Create an Optuna study and optimize the objective
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = n_trials)

    # Extract the best parameters and model
    tuned_params = study.best_params
    print('The best parameters are:', tuned_params)
    
    tuned_model = RandomForestClassifier(n_estimators = tuned_params['n_estimators'], 
                                        max_depth = tuned_params['max_depth'],
                                        min_samples_split = tuned_params['min_samples_split'], 
                                        min_samples_leaf = tuned_params['min_samples_leaf'],
                                        class_weight = 'balanced', 
                                        random_state=0)
    
    # Trains the model with the best hyperparameters
    tuned_model.fit(X_train, y_train)

    return tuned_model

def plot_importance(model, X_train):

    """
    Plot feature importances of the model.

    Paramters: 
    - model: model object to plot
    - X_train: train set
    """

    importances = np.round(model.feature_importances_ / model.feature_importances_.sum(), 4)
    feature_importances = pd.DataFrame({
    'features': X_train.columns,
    'importance': importances}).sort_values(by='importance', ascending=True).reset_index()

    fig = px.bar(feature_importances, 
                x = 'importance', 
                y = 'features', 
                orientation='h', 
                height = 650,
                labels={'importance': 'Importance', 'features': 'Feature'}, 
                title = 'Feature Importance')
    
    fig.update_traces(marker_color='#10b38f')
    
    fig.update_layout(margin = dict(t = 40, b = 30),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')

    return fig

def powerco():
    return ('''
            Powerco is a major gas and electricity provider that supplies corporate, small and medium enterprises (SME), and residential customers. 
            The power liberalization of the energy market in Europe has led to significant customer churn, especially in the SME segment. 
            ''')

def explain_churn():
    return ('''
            Customer churn is a silent profit killer for companies. 
            Losing existing customers not only lowers the immediate revenue but also increases costs significantly, 
            as acquiring new customers can be 5 to 25 times more expensive than retaining current ones.
            \n
            By identifying churn trends, companies can enhance customer satisfaction and loyalty to ultimately boost profitability.
            The savings generated by addressing customer churn can be invested into growth or innovation.
            ''')

def pricing_strategies():
    return ('''
            Pricing strategies of electricity companies are designed to cover costs, ensure profitability, and remain attractive to customers. 
            Key aspects of electricity pricing include:

            - **Peak and Off-Peak Pricing:** Price rates vary depending on demand. Electricity is often more expensive during high-demand periods 
                (peak hours, e.g., weekday mornings and evenings) and cheaper during low-demand times (off-peak hours, e.g., at night).
            - **Variable and Fixed Pricing:** Some price rates are variable since they depend on market prices, while fixed prices include 
                connection fees or base service charges.
            
            For this churn analysis, fixed prices are omitted because customers are more likely to react to changes in variable costs, 
            which fluctuate with market conditions and ultimately reflect their consumption patterns.
            ''')

def price_sensitivity():
    return ('''
            Intuitively, churn is often linked to price sensitivity. **Price sensitivity** is a qualitative concept that refers to how 
            aware and reactive customers are to price changes. **Price elasticity** quantitatively measures how the demand changes in 
            response to price changes. Higher average prices, price ratios, and ranges, along with greater volatility, can signal financial strain and instability, which may contribute to churn.
             
            **Hypothesis:**
            - **H₀**: Price features have no significant impact on churn.
            - **H₁**: Price features significantly impact churn.
            ''')
                    
def pricing_features():
    return ('''
            - **Average Prices:** Typical costs customers face during a time period.
            - **Standard Deviations:** Variability in prices over time (volatility) during a time period.
            - **Price Ranges:** Difference between maximum and minimum prices during a time period.
            - **Price Ratios:** Price imbalance between different time periods.
             ''')

def dataset_overview():
    return ('''
             There are two datasets—one for the client data and one for the price data. 
             These datasets are merged, and additional contractual, consumptional, and pricing features are engineered to provide insights into customer behavior and service usage.

            - **Customer**: Identifier (32-character string)
            - **Contract Features:** Customer tenure and contract-related timelines such as months remaining, time since last renewal, or modification of the product.
            - **Consumption Features:** Historical and forecasted electricity and gas usage.
            - **Pricing Features:** Average, standard deviation, and range of electricity prices during different time periods, as well as price ratios across different time periods.
            - **Churn:** Whether a customer has left PowerCo or remained.
            ''')

def feature_descriptions():
    return ('''
             - **id:** Client company identifier (32-character string).
             - **channel_sales:** Identifier of the sales channel (cateories 0-4).
             - **tenure_years:** Client's tenure in years.
             - **months_to_end:** Months remaining until the end of the current contract.
             - **months_since_modified:** Months since the product was last modified.
             - **months_since_renewed:** Months since the contract was last renewed.
             - **yearly_consumption:** Total electricity consumption in the past year (kWh).
             - **monthly_consumption:** Electricity consumption in the last month (kWh).
             - **yearly_gas_consumption:** Total gas consumption in the past year (for clients with gas service).
             - **yearly_forecast_consumption:** Forecasted electricity consumption for the next year (kWh).
             - **yearly_forecast_meter_rent:** Forecasted meter rental costs for the next year (€).
             - **paid_consumption:** Billed electricity consumption (kWh).
             - **net_maargin_electricity:** Profit margin from electricity usage (€).
             - **net_margin_power:** Profit margin from power usage (€).
             - **net_margin:** Total net profit margin across all services (€).
             - **avg_price_off_peak_var:** Average variable price for electricity during off-peak hours (€).
             - **avg_price_peak_var:** Average variable price for electricity during peak hours (€).
             - **avg_price_mid_peak_var:** Average variable price for electricity during mid-peak hours (€).
             - **price_peak_off_peak_ratio:** Ratio of peak-hour prices to off-peak prices.
             - **std_price_off_peak_var:** Standard deviation of variable prices during off-peak hours.
             - **std_price_peak_var:** Standard deviation of variable prices during peak hours.
             - **std_price_mid_peak_var:** Standard deviation of variable prices during mid-peak hours.
             - **range_price_peak_var:** Price range for peak-hour variable prices.
             - **range_price_off_peak_var:** Price range for off-peak variable prices.
             - **range_price_mid_peak_var:** Price range for mid-peak variable prices.
             - **churn:** Binary indicator of whether the client has churned (1 = churned, 0 = retained).
            ''')

def explain_forest():
    return ('''
        The Random Forest is an ensemble machine learning algorithm used for classification and regression tasks. 
        It builds multiple decision trees during training and combines their outputs to make more accurate and robust predictions compared to a single decision tree.
        
        Key features are:
        - **Bagging:** Trains each tree on a random subset of the data to reduce overfitting.
        - **Feature Randomness:** Considers random subsets of features at each split to lower the correlation between trees.
        - **Ensemble Learning:** Aggregates predictions (majority vote) from multiple trees to improve accuracy and stability.
        ''')

def explain_hyperparameters():
    return ('''
            - **Number of Trees in the Forest:** 
            Too few trees might underfit the data, while too many trees can lead to unnecessary computation (Default = 100).
            
            - **Maximum Tree Depth:** 
            Determines how much the model learns from the data. Deeper trees can capture more complex patterns but are prone to overfitting. 
            Per default all nodes are expanded until all leaves are pure or contain less than the minimum required samples of a splits.
            
            - **Minimum Samples of a Split:** 
            Controls if an internal node can be split given the remaining number of samples. 
            Larger values prevent overly deep trees and reduce overfitting (Default = 2).
            
            - **Minimum Samples of a Leaf:** 
            Ensures that leaf nodes have a minimum number of samples. 
            This prevents the model from being overly sensitive to individual data points and therefore smooth decision boundaries (Default = 1).
            ''')

def plot_conclusions():
    return ('''
            The churn and no-churn distributions show noticeable differences, yet they do not exhibit clear linear patterns. 
            This suggests that the features likely have complex, non-linear relationships. Therefore, it is essential for the machine learning algorithm 
            to take the majority of columns into account to capture any significant correlations.
            ''')

def explain_metrics():
    return ('''
            In churn analysis, **accuracy** is often insufficient for evaluating model performance, particularly with imbalanced datasets where most customers do not churn. 
            Accuracy misses the model's ability to detect the minority class (churners), which is critical. 
            From a business perspective, false negatives (missed churners) are far more costly than false positives. 
            Thus, optimizing for **recall** is key, as it ensures to capture as many churners as possible.

            Although an accuracy of around 90% is achieved for nearly every combination of features, 
            this is not meaningful when precision and recall are low.
            ''')

def explain_bayesian():
    return ('''
            **Bayesian optimization** uses past evaluations to estimate which hyperparameter regions are promising and explores them more. 
            Meanwhile it balances exploration (trying new regions) and exploitation (focusing on areas with good results). 
            It works well with continuous, large, or high-dimensional datasets like this one.
            ''')

def explain_performance():
    return('''
           ''')

def final_conclusions():
    return('''
            **Performance of the Random Forest Model**
               
            Through hyperparameter tuning, an accuracy of 83%, precision of 23%, and recall of 31% have been achieved. 
            While the accuracy seems relatively promising, the recall value is still too low to effectively identify churners and minimize the costs associated with missed churners.

            **Feature Importance**
           
            Efforts to improve the model have included hyperparameter tuning and dataset balancing. 
            Despite these attempts, the recall remains low. However, the feature importance plot clearly identifies which features correlate most with churn. 
            Consumptive and margin-related features are most influential, while contract-related features also seem to have some impact, although to a lesser degree. 
            
            Intuitively, clients on long-term or previously modified contracts might not exhibit the same churn behavior as those on shorter-term and unchanged plans.
            High consumption often translates to higher bills and higher provider margins, which can make these customers more sensitive to price changes or competitor offers.
            Interestingly, the engineered price-related features have a superior influence on churn predictions, underscoring that consumption and price are strongly linked. 
            To target these high-consumption customers, the company could offer tailored discounts or bonuses to retain them and reduce churn in this valuable segment.

            **Error Sources**
           
            One potential source of error in the current model is the inclusion of too many features. Irrelevant features can confuse the model, leading to overfitting. 
            To address this, a more refined feature selection should be employed, omitting features that provide redundant or overlapping information. 
            Additionally, creating other meaningful features could help the model focus on the most significant predictors of churn.

            **Alternatives**
           
            Ultimately, while the random forest model has shown some potential, it may not be the most effective algorithm for this problem. 
            Exploring alternative machine learning algorithms, such as XGBoost, which often perform better with imbalanced datasets and complex relationships, 
            could result in more accurate predictions. This, in turn, could lead to significant savings for PowerCo.
             ''')