import streamlit as st
import pandas as pd
import plotly.express as px
from functions import load_from_pickle, save_to_pickle, plot_categorical, plot_numerical, plot_kde
from functions import dummies, prepare_data, train_random_forest, evaluate_model, plot_importance 
from functions import powerco, pricing_strategies, price_sensitivity, pricing_features, explain_churn, dataset_overview
from functions import feature_descriptions, explain_forest, explain_hyperparameters, plot_conclusions, explain_metrics, explain_bayesian, explain_performance, final_conclusions

def main():
    
    st.title("Modeling Churn")
    
    #st.image("../data/PowerCo.jpg", width = 300)
    st.subheader('PowerCo')
    st.write(powerco())

    with st.expander("Expand for Explanation of Pricing Strategies of Electricity Providers"):
        st.subheader('Pricing Strategies')
        st.write(pricing_strategies())

    st.write("**This interactive dashboard outlines the process for diagnosing churn and implementing a predictive model to anticipate churn of PowerCo's clients.**")

    url = 'https://www.kaggle.com/datasets/erolmasimov/powerco/data'
    st.write("Data source: [link](%s)" % url)

    st.subheader('Why Identifying Churn is Crucial')
    st.write(explain_churn())
    
    st.subheader('Dataset')
    st.write(dataset_overview())
    
    with st.expander("Expand for Explanation of Pricing Features"):
        st.write(pricing_features())

    df = load_from_pickle('../pickle/describe_price.pkl')
    st.dataframe(df.head(5))

    with st.expander("Expand for Detailed Feature Descriptions"):
        st.write(feature_descriptions())
    
    st.markdown('<p style="color:#ff8379; font-size: 16px; font-weight: bold;">The analysis of the PowerCo dataset reveals a churn rate of approximately 10%, indicating a notable portion of customers are leaving the service.</p>', unsafe_allow_html=True)
    
    st.subheader('Churn Distribution')
    st.write('''
            Use the filter to select a column and explore the relationship between each feature and customer churn. 
            For numerical columns, a KDE plot is also displayed. 
            This step helps for feature selection and visually substantiates the development of the predictive model.
            ''')
    
    st.sidebar.header("Churn Distribution")
    categorical_columns = ['tenure_years', 'channel_sales']
    numerical_columns = ['months_to_end', 'months_since_modified', 'months_since_renewed', 
                         'yearly_consumption', 'monthly_consumption', 'yearly_forecast_consumption', 
                         'yearly_forecast_meter_rent', 'paid_consumption', 'net_margin_electricity',
                         'net_margin_power', 'net_margin', 
                         'avg_price_off_peak_var', 'avg_price_peak_var', 'avg_price_mid_peak_var',
                         'price_peak_off_peak_ratio_var', 'std_price_off_peak_var', 'std_price_peak_var']
    selected_column = st.sidebar.selectbox("Select Column", categorical_columns + numerical_columns)

    # Display plot for the selected column
    if selected_column:
        
        if selected_column in categorical_columns:
            fig = plot_categorical(df, selected_column)
            st.plotly_chart(fig)
        
        elif selected_column in numerical_columns:
            
            min_value = float(df[selected_column].min())  # Minimum value in the column
            max_value = float(df[selected_column].max())

            kde_min, kde_max = st.sidebar.slider(
            "Select KDE Range",
            min_value=min_value, max_value=max_value,
            value=(min_value, max_value),step=(max_value - min_value) / 100)
            
            fig1 = plot_numerical(df, selected_column, bins = 12)
            fig2 = plot_kde(df, selected_column, min = kde_min, max = kde_max)

            st.plotly_chart(fig1)
            st.plotly_chart(fig2)

    st.write('**Conclusions**')
    st.write(plot_conclusions())

    st.title("Random Forest Algorithm")
    st.sidebar.subheader("Random Forest Classifier")

    with st.expander("Expand for Explanation of the Random Forest Algorithm"):
        st.image("../plots/Random-Forest-Algorithm.png", use_container_width = True)
        st.subheader('What is a Random Forest Model?')
        st.write(explain_forest())
    
    st.write('''
    **Why is the Random Forest Algorithm Ideal for Churn Analysis?**
        
    The random forest classifier is an optimal choice due to its ability to handle complex, non-linear relationships and its robustness against overfitting. 
    It effectively identifies important features within high-dimensional datasets and provides high accuracy, even in imbalanced scenarios (10% churn).     
    ''')

    st.subheader('Develop a Custom Random Forest Model')
    st.write('''This application allows to implement a custom random forest model by choosing a set of features 
             and adjust the key hyperparameters. The algorithm already considers the imbalance of the dataset.
             ''')
    
    with st.expander("Expand for Explanation of Hyperparameters"):
        st.write(explain_hyperparameters())

    df = dummies(df, ['channel_sales'])

    select_all = st.sidebar.checkbox("Select All Columns")
    options = df.columns.difference(['churn', 'id'])

    if select_all:
        included_columns = list(options)
    else:
        included_columns = st.sidebar.multiselect("Select Columns to Include", options=options)

    data_cache_path = "../pickle/prepared_data.pkl"
    model_cache_path = "../pickle/model.pkl"
    tuned_model_path = "../pickle/tuned_model.pkl"

    if included_columns:
        
        if "prev_columns" not in st.session_state:
            st.session_state.prev_columns = None

        if "hyperparameters" not in st.session_state:
            st.session_state.hyperparameters = {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 2,
                "min_samples_leaf": 1}
        
        if "prev_hyperparameters" not in st.session_state:
            st.session_state.prev_hyperparameters = None

        st.sidebar.header("Hyperparameters")
        st.session_state.hyperparameters["n_estimators"] = st.sidebar.slider(
            "Number of Trees", min_value=10, max_value=500,
            value=st.session_state.hyperparameters["n_estimators"], step = 10)
        st.session_state.hyperparameters["max_depth"] = st.sidebar.slider(
            "Maximum Tree Depth", min_value=1, max_value=50, 
            value=st.session_state.hyperparameters["max_depth"], step = 1)
        st.session_state.hyperparameters["min_samples_split"] = st.sidebar.slider(
            "Minimum Samples of a Split", min_value=2, max_value=20, 
            value=st.session_state.hyperparameters["min_samples_split"], step = 1)
        st.session_state.hyperparameters["min_samples_leaf"] = st.sidebar.slider(
            "Minimum Samples of a Leaf", min_value=1, max_value=3, 
            value=st.session_state.hyperparameters["min_samples_leaf"], step = 1)
        
        prev_columns = st.session_state.prev_columns
        prev_hyperparameters = st.session_state.prev_hyperparameters

        if prev_columns != included_columns or prev_hyperparameters != st.session_state.hyperparameters:

            X_train, X_test, y_train, y_test, scaler = prepare_data(df, df.columns.difference(included_columns).tolist(), 'churn')
            save_to_pickle((X_train, X_test, y_train, y_test, scaler), data_cache_path)

            model = train_random_forest(X_train, y_train,
                                    n_estimators=st.session_state.hyperparameters["n_estimators"],
                                    max_depth=st.session_state.hyperparameters["max_depth"],
                                    min_samples_split=st.session_state.hyperparameters["min_samples_split"],
                                    min_samples_leaf=st.session_state.hyperparameters["min_samples_leaf"])

            save_to_pickle(model, model_cache_path)
            st.session_state.prev_columns = included_columns
            st.session_state.prev_hyperparameters = st.session_state.hyperparameters.copy()

        else: 
            model = load_from_pickle(model_cache_path)
            X_train, X_test, y_train, y_test, scaler = load_from_pickle(data_cache_path)

        st.write('**Performance Evaluation**')
        st.write('The model is evaluated by calculating accuracy, precision, and recall.')
        with st.expander("Expand for Explanation of Performance Metrics"):
            st.write(explain_performance())
        st.write(explain_metrics())

        scores = evaluate_model(model, X_test, y_test)
        st.plotly_chart(scores)
    
    else:
        st.warning("Please select columns to train the model.")

    st.header('Hyperparameter Tuning')
    st.write('**Bayesian optimization** is used to fine-tune the model by considering all features and maximizing for recall.')
    with st.expander("Expand for Explanation of Bayesian Optimization"):
        st.write(explain_bayesian())
        
    if st.checkbox("View Results of Hyperparameter Tuning"):
        tuned_model = load_from_pickle(tuned_model_path)
        best_params = {"Number of Trees": tuned_model.n_estimators,
                        "Maximum Tree Depth": tuned_model.max_depth,
                        "Minimum Samples of a Split": tuned_model.min_samples_split,
                        "Minimum Samples of a Leaf": tuned_model.min_samples_leaf}

        best_params_df = pd.DataFrame([best_params])
        st.dataframe(best_params_df.reset_index(drop = True))

        X_train_full, X_test_full, y_train, y_test, scaler = prepare_data(df, ['id', 'churn'], 'churn')
        
        tuned_scores = evaluate_model(tuned_model, X_test_full, y_test)
        st.plotly_chart(tuned_scores)

        tuned_importance = plot_importance(tuned_model, X_train_full)
        st.plotly_chart(tuned_importance)
        st.write('Feature selection helps to identify the features that are most strongly correlated with the target variable.')

        st.subheader('Conclusions')
        st.write(final_conclusions())

    
if __name__ == "__main__":
    main()


