# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:59:33 2024

@author: Fik
"""

import streamlit as st

import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def preprocess_data(df):
    df = df.drop(['id', 'regional_unit', 'municipality'], axis=1)
    # Convert to date-time 'start_production_date' column
    df['start_production_date'] = pd.to_datetime(df['start_production_date'])

    district_dict = {'ΗΠΕΙΡΟΥ': 'Epirus', 'ΗΠΕΙΡΟΥ ': 'Epirus', 'ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ': 'Cent_Macedonia',
                     'ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ ': 'Cent_Macedonia', 'Κ ΜΑΚΕΔΟΝΙΑΣ': 'Cent_Macedonia',
                     'ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ': 'West_Macedonia', 'Δ ΜΑΚΕΔΟΝΙΑΣ': 'West_Macedonia',
                     'ΙΟΝΙΩΝ ΝΗΣΙΩΝ': 'Ionian_Islands', 'ΙΟΝΙΩΝ ΝΗΣΩΝ': 'Ionian_Islands',
                     'ΣΤΕΡΕΑΣ ΕΛΛΑΔΟΣ': 'Cent_Greece', 'ΣΤΕΡΕΑΣ ΕΛΛΑΔΟΣ ': 'Cent_Greece',
                     'ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ': 'Cent_Greece', 'ΣΤ ΕΛΛΑΔΑΣ': 'Cent_Greece', 'ΚΡΗΤΗΣ': 'Crete',
                     'ΒΟΡΕΙΟΥ ΑΙΓΑΙΟΥ': 'North_Aegean', 'ΘΕΣΣΑΛΙΑΣ': 'Thessaly', 'ΝΟΤΙΟΥ ΑΙΓΑΙΟΥ': 'South_Aegean',
                     'ΔΥΤΙΚΗΣ ΕΛΛΑΔΟΣ': 'West_Greece', 'Δ ΕΛΛΑΔΑΣ': 'West_Greece',
                     'ΔΥΤΙΚΗΣ ΕΛΛΑΔΟΣ ': 'West_Greece', 'ΔΥΤΙΚΗΣ ΕΛΛΑΔΑΣ': 'West_Greece', 'ΠΕΛΟΠΟΝΝΗΣΟΥ': 'Peloponnese',
                     'ΠΕΛΟΠΟΝΝΗΣΟΥ             ': 'Peloponnese', 'ΑΤΤΙΚΗΣ': 'Attica',
                     'ΑΤΤΙΚΗΣ ': 'Attica', 'ΑΝΑΤΟΛΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ ΚΑΙ ΘΡΑΚΗΣ': 'East_Macedonia_Thrace',
                     'ΑΝΑΤΟΛΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ & ΘΡΑΚΗΣ': 'East_Macedonia_Thrace',
                     'ΑΝ ΜΑΚΕΔΟΝΙΑΣ ΘΡΑΚΗΣ': 'East_Macedonia_Thrace',
                     'ΑΝ. ΜΑΚΕΔΟΝΙΑΣ & ΘΡΑΚΗΣ': 'East_Macedonia_Thrace',
                     'ΣΤΕΡΕΑΣ ΕΛΛΑΔΟΣ - ΘΕΣΣΑΛΙΑΣ': 'Cent_Greece_Thessaly',
                     'ΣΤΕΡΕΑΣ ΕΛΛΑΔΟΣ & ΘΕΣΣΑΛΙΑΣ': 'Cent_Greece_Thessaly',
                     'ΘΕΣΣΑΛΙΑΣ,ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ': 'Cent_Greece_Thessaly',
                     'ΔΥΤΙΚΗΣ ΕΛΛΑΔΟΣ -  ΚΕΝΤΡΙΚΗΣ  ΜΑΚΕΔΟΝΙΑΣ': 'West_Greece_Cent_Macedonia',
                     'ΘΕΣΣΑΛΙΑΣ & ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ': 'Thessaly_West_Macedonia',
                     'ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ,ΘΕΣΣΑΛΙΑΣ': 'Thessaly_West_Macedonia',
                     'ΘΕΣΣΑΛΙΑΣ,ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ': 'Thessaly_West_Macedonia',
                     'ΔΥΤΙΚΗΣ ΕΛΛΑΔΟΣ & ΣΤΕΡΕΑΣ ΕΛΛΑΔΟΣ': 'West_Greece_Cent_Greece',
                     'ΣΤΕΡΕΑΣ ΕΛΛΑΔΟΣ  -  ΑΤΤΙΚΗΣ': 'Attica_Cent_Greece',
                     'ΑΤΤΙΚΗΣ,ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ': 'Attica_Cent_Greece',
                     'ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ,ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ': 'West_Macedonia_Cent_Macedonia',
                     'ΘΕΣΣΑΛΙΑΣ,ΗΠΕΙΡΟΥ': 'Thessaly_Epirus',
                     'ΑΝ. ΜΑΚΕΔΟΝΙΑΣ & ΘΡΑΚΗΣ,ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ': "East_Macedonia_Thrace_Central_Macedonia"
                     }
    # Map district values using district_dict
    df['district'] = df['district'].map(district_dict)

    categorical_columns = ['district', 'tech']
    # Convert categorical columns to dummy variables (1-0 encoding)
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    # Convert boolean columns to integers (1-0)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    return df


def transform_data_on_time_level(df, cols_to_drop_list, start_time_col, time_period):
    date_column = 'start_production_date'
    df_period = df.drop(cols_to_drop_list, axis=1)

    if time_period == 'M':
        # Convert start_production_month to datetime (start of the month)
        df_period[start_time_col] = pd.to_datetime(df_period[date_column].dt.to_period('M').astype(str) + '-01')
    elif time_period == 'W':
        # Convert start_production_week to datetime (start of the week)
        df_period[start_time_col] = df_period[date_column].dt.to_period(time_period).apply(lambda r: r.start_time)

    df_period = df_period.drop([date_column], axis=1)
    df_period['num_permits'] = 1

    # Define the aggregation functions
    aggregation_funcs = {col: 'mean' for col in df_period.columns if
                         col not in [start_time_col, 'max_mp', 'num_permits']}
    aggregation_funcs['max_mp'] = 'sum'
    aggregation_funcs['num_permits'] = 'sum'

    # Perform the aggregation
    aggr = df_period.groupby(start_time_col).agg(aggregation_funcs).reset_index()

    # Rename aggregated columns to include 'avg' in their names
    aggr.rename(columns={col: f'avg_{col}' for col in aggregation_funcs if
                         col not in [start_time_col, 'max_mp', 'num_permits']}, inplace=True)

    # Rename specific columns to desired names
    aggr.rename(columns={'max_mp': 'total_mw'}, inplace=True)

    return aggr


def sarima_model_train(train_data, target_var, freq):
    # Determine the seasonal period
    if freq == 'W':
        seasonal_period = 52  # Weekly data
    elif freq == 'M':
        seasonal_period = 12  # Monthly data

    # Fit a SARIMA Model
    model = SARIMAX(train_data[target_var],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, seasonal_period))

    results = model.fit(disp=False)
    print(results.summary())
    return results


def sarima_model(train_data, k, target_var, freq):
    results = sarima_model_train(train_data, target_var, freq)
    predictions = results.forecast(steps=k)

    return predictions


def create_lagged_features(df, target_var, lags):
    for lag in range(1, lags + 1):
        df[f'{target_var}_lag{lag}'] = df[target_var].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


def random_forest_model(train_data, test_data, target_var, time_col):
    X = train_data.drop(columns=[target_var, time_col])
    y = train_data[target_var]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    X_test = test_data.drop(columns=[target_var, time_col])
    predictions = model.predict(X_test)
    return predictions


def xgboost_model(train_data, test_data, target_var, time_col):
    X = train_data.drop(columns=[target_var, time_col])
    y = train_data[target_var]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X, y)
    X_test = test_data.drop(columns=[target_var, time_col])
    predictions = model.predict(X_test)

    return predictions


def prophet_model_train(df, target_var, time_col):
    prophet_df = df[[time_col, target_var]].rename(columns={time_col: 'ds', target_var: 'y'})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    return model


def prophet_model(df, k, target_var, time_col):
    model = prophet_model_train(df, target_var, time_col)
    future = model.make_future_dataframe(periods=k, include_history=False)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(k)


def evaluate_model(test_data, predictions, target_var):
    mse = mean_squared_error(test_data[target_var], predictions)
    r2 = r2_score(test_data[target_var], predictions)
    st.write(f'MSE: {mse}')
    st.write(f'R^2: {r2}')


def plot_forecast(df, predictions, train_data, k, target_var, time_col, freq):
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_col], df[target_var], label='Historical Data')
    future_dates = pd.date_range(train_data[time_col].iloc[-1], periods=len(predictions) + 1, freq=freq)[1:]
    plt.plot(future_dates, predictions, label='Forecasted Data', color='red')
    plt.xlabel('Date')
    plt.ylabel(target_var)

    if freq == 'W':
        seasonal_period = ' Weeks'
    elif freq == 'M':
        seasonal_period = ' Months'

    plt.title('Permit Forecast for the Next ' + str(k) + seasonal_period)
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    plt.close()


def predict_with_dif_models(hist_data, k, target_var, time_col, freq, lags=4):
    models_predictions_list = []

    train_data = hist_data.iloc[:-k]
    test_data = hist_data.iloc[-k:]

    hist_data_lagged = create_lagged_features(hist_data, target_var, lags)
    train_data_lagged = hist_data_lagged.iloc[:-k]
    test_data_lagged = hist_data_lagged.iloc[-k:]

    # Forecast for the next k periods
    sarima_predictions = sarima_model(train_data, k, target_var, freq)
    models_predictions_list.append(('SARIMA', sarima_predictions))

    rf_predictions = random_forest_model(train_data_lagged, test_data_lagged, target_var, time_col)
    models_predictions_list.append(('Random Forest', rf_predictions))

    xgb_predictions = xgboost_model(train_data_lagged, test_data_lagged, target_var, time_col)
    models_predictions_list.append(('XGBoost', xgb_predictions))

    prophet_predictions = prophet_model(hist_data, k, target_var, time_col)
    models_predictions_list.append(('Prophet', prophet_predictions['yhat'].values))

    for model_name, predictions in models_predictions_list:
        st.write(f"### {model_name} Predictions")
        # Round the predictions to the nearest integer
        if target_var == 'num_permits':
            predictions = predictions.round()

        #st.write("Predictions:", predictions)
        # Evaluate the model
        evaluate_model(test_data, predictions, target_var)

        # Plot the forecast
        plot_forecast(hist_data, predictions, train_data, k, target_var, time_col, freq)


def main():
    pd.set_option("display.max.columns", None)

    df = pd.read_csv('G:/DWS/IWW-02-07/project 02/final_permits.csv')
    df = preprocess_data(df)

    st.title("Permit Forecasting App")
    #st.write("Data Shape:", df.shape)

    freq = st.selectbox('Select Frequency', ('Weekly', 'Monthly'))
    if freq == 'Weekly':
        cols_to_drop_list = ['start_production_month', 'approval_period_in_months', 'approved_time_in_months',
                             'monthly_co2_price']
        time_col = 'start_production_week'
        time_period = 'W'
    else:
        cols_to_drop_list = ['start_production_week', 'approval_period_in_weeks', 'approved_time_in_weeks',
                             'weekly_co2_price']
        time_col = 'start_production_month'
        time_period = 'M'

    target_var = st.selectbox('Select Target Variable', ('num_permits', 'total_mw'))
    k = st.slider('Select number of periods to forecast', 1, 50, 20)
    lags = st.slider('Select number of lags', 1, 20, 4)

    df_aggr = transform_data_on_time_level(df, cols_to_drop_list, time_col, time_period)

    if st.button('Predict'):
        predict_with_dif_models(df_aggr, k, target_var, time_col, time_period, lags)


if __name__ == '__main__':
    main()
