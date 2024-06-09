from flask import Flask, request
import pandas as pd
import forecasting_models as fm
from json import loads, dumps

app = Flask(__name__)


@app.route('/prophet/month/num_permits', methods=['GET'])
def prophet_month_num_permits():
    k = int(request.args['future'])
    future = prophet_model_month_num_permits.make_future_dataframe(periods=k, include_history=False)
    forecast = prophet_model_month_num_permits.predict(future)
    predictions = forecast['yhat'].values
    return predictions.tolist()


@app.route('/prophet/month/total_mw', methods=['GET'])
def prophet_month_total_mw():
    k = int(request.args['future'])
    future = prophet_model_month_total_mw.make_future_dataframe(periods=k, include_history=False)
    forecast = prophet_model_month_total_mw.predict(future)
    predictions = forecast['yhat'].values
    return predictions.tolist()


@app.route('/prophet/week/num_permits', methods=['GET'])
def prophet_week_num_permits():
    k = int(request.args['future'])
    future = prophet_model_week_num_permits.make_future_dataframe(periods=k, include_history=False)
    forecast = prophet_model_week_num_permits.predict(future)
    predictions = forecast['yhat'].values
    return predictions.tolist()


@app.route('/prophet/week/total_mw', methods=['GET'])
def prophet_week_total_mw():
    k = int(request.args['future'])
    future = prophet_model_week_total_mw.make_future_dataframe(periods=k, include_history=False)
    forecast = prophet_model_week_total_mw.predict(future)
    predictions = forecast['yhat'].values
    return predictions.tolist()


@app.route('/sarima/month/num_permits', methods=['GET'])
def sarima_month_num_permits():
    k = int(request.args['future'])
    forecast = sarima_model_month_num_permits.forecast(steps=k)
    return loads(forecast.to_json(orient='records'))


@app.route('/sarima/month/total_mw', methods=['GET'])
def sarima_month_total_mw():
    k = int(request.args['future'])
    forecast = sarima_model_month_total_mw.forecast(steps=k)
    return loads(forecast.to_json(orient='records'))


@app.route('/sarima/week/num_permits', methods=['GET'])
def sarima_week_num_permits():
    k = int(request.args['future'])
    forecast = sarima_model_week_num_permits.forecast(steps=k)
    return loads(forecast.to_json(orient='records'))


@app.route('/sarima/week/total_mw', methods=['GET'])
def sarima_week_total_mw():
    k = int(request.args['future'])
    forecast = sarima_model_week_total_mw.forecast(steps=k)
    return loads(forecast.to_json(orient='records'))


if __name__ == '__main__':
    df = pd.read_csv('results\\final_permits.csv')
    df = fm.preprocess_data(df)

    cols_to_drop_list_month = ['start_production_week', 'approval_period_in_weeks', 'approved_time_in_weeks', 'weekly_co2_price']
    df_train_month = fm.transform_data_on_time_level(df, cols_to_drop_list_month, 'start_production_month', 'M')

    cols_to_drop_list_week = ['start_production_month', 'approval_period_in_months', 'approved_time_in_months', 'monthly_co2_price']
    df_train_week = fm.transform_data_on_time_level(df, cols_to_drop_list_week, 'start_production_week', 'W')

    sarima_model_month_num_permits = fm.sarima_model_train(df_train_month, 'num_permits', 'M')
    prophet_model_month_num_permits = fm.prophet_model_train(df_train_month, 'num_permits', 'start_production_month')

    sarima_model_month_total_mw = fm.sarima_model_train(df_train_month, 'total_mw', 'M')
    prophet_model_month_total_mw = fm.prophet_model_train(df_train_month, 'total_mw', 'start_production_month')

    sarima_model_week_num_permits = fm.sarima_model_train(df_train_week, 'num_permits', 'W')
    prophet_model_week_num_permits = fm.prophet_model_train(df_train_week, 'num_permits', 'start_production_week')

    sarima_model_week_total_mw = fm.sarima_model_train(df_train_week, 'total_mw', 'W')
    prophet_model_week_total_mw = fm.prophet_model_train(df_train_week, 'total_mw', 'start_production_week')

    app.run()
