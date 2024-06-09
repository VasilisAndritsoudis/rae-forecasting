import streamlit as st
import pandas as pd
import plotly.express as px
import forecasting_models as fm
import requests


def load_df():
    return pd.read_csv('results\\permits.csv')


def visualize_timeseries(df: pd.DataFrame, period_options, attribute_type_options, timeseries_type_options):
    column = ''
    co2_column = ''
    if period_options == 'Μήνας':
        column = 'ΜΗΝΑΣ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'
        co2_column = 'ΜΗΝΙΑΙΑ ΤΙΜΗ ΔΙΟΞΕΙΔΙΟΥ ΤΟΥ ΑΝΘΡΑΚΑ'
    elif period_options == 'Εβδομάδα':
        column = 'ΕΒΔΟΜΑΔΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'
        co2_column = 'ΕΒΔΟΜΑΔΙΑΙΑ ΤΙΜΗ ΔΙΟΞΕΙΔΙΟΥ ΤΟΥ ΑΝΘΡΑΚΑ'

    st_df = pd.DataFrame()
    if 'Α.Π.Ε.' in timeseries_type_options and 'Τιμή CO2' in timeseries_type_options:
        rae_df = pd.DataFrame()
        if 'Μέγιστη Ισχύς (MW)' in attribute_type_options:
            rae_df = df[[column, 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby(column).sum().reset_index()
        elif 'Αριθμός Α.Π.Ε' in attribute_type_options:
            rae_df = df.groupby(column).size().reset_index()

        co2_tf = df[[column, co2_column]].groupby(column).sum().reset_index()
        st_df = rae_df.merge(co2_tf, on=column)
    elif 'Α.Π.Ε.' in timeseries_type_options:
        if 'Μέγιστη Ισχύς (MW)' in attribute_type_options:
            st_df = df[[column, 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby(column).sum().reset_index()
        elif 'Αριθμός Α.Π.Ε' in attribute_type_options:
            st_df = df.groupby(column).size().reset_index()
    elif 'Τιμή CO2' in timeseries_type_options:
        st_df = df[[column, co2_column]].groupby(column).sum().reset_index()

    st_df = st_df.rename(columns={column: 'index'}).set_index('index')
    st.line_chart(st_df)


def visualize_data_per_region(df: pd.DataFrame, attribute_type_options, region_options):
    technologies = df['ΤΕΧΝΟΛΟΓΙΑ'].unique()
    tech_values = {}
    for tech in technologies:
        tech_values[tech] = []

    df = df[df['ΠΕΡΙΦΕΡΕΙΑ'].isin(region_options)]
    region_options.sort()

    if 'Μέγιστη Ισχύς (MW)' in attribute_type_options:
        st_df = df[['ΠΕΡΙΦΕΡΕΙΑ', 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby('ΠΕΡΙΦΕΡΕΙΑ').sum().reset_index()
        for region in region_options:
            if region not in st_df['ΠΕΡΙΦΕΡΕΙΑ'].tolist():
                st_df.loc[-1] = [region, 0]

        st_df = st_df.sort_values('ΠΕΡΙΦΕΡΕΙΑ')

        for region in region_options:
            tmp_df = df.loc[df['ΠΕΡΙΦΕΡΕΙΑ'] == region]
            for tech in technologies:
                tmp2_df = tmp_df.loc[tmp_df['ΤΕΧΝΟΛΟΓΙΑ'] == tech]
                tmp2_df = tmp2_df[['ΤΕΧΝΟΛΟΓΙΑ', 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby('ΤΕΧΝΟΛΟΓΙΑ').sum().reset_index()
                if len(tmp2_df['ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)'] > 0):
                    tech_values[tech].append(tmp2_df['ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)'][0])
                else:
                    tech_values[tech].append(0)

        for tech in technologies:
            st_df[tech] = tech_values[tech]

        st_df = st_df.rename(columns={'ΠΕΡΙΦΕΡΕΙΑ': 'index'}).set_index('index')
        st.table(st_df)
        st_df.drop('ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)', axis=1, inplace=True)
        st.bar_chart(st_df)
    elif 'Αριθμός Α.Π.Ε' in attribute_type_options:
        st_df = df[['ΠΕΡΙΦΕΡΕΙΑ']].groupby('ΠΕΡΙΦΕΡΕΙΑ').size().to_frame('ΣΥΝΟΛΟ').reset_index()
        for region in region_options:
            if region not in st_df['ΠΕΡΙΦΕΡΕΙΑ'].tolist():
                st_df.loc[-1] = [region, 0]

        st_df = st_df.sort_values('ΠΕΡΙΦΕΡΕΙΑ')

        for region in region_options:
            tmp_df = df.loc[df['ΠΕΡΙΦΕΡΕΙΑ'] == region]
            for tech in technologies:
                tmp2_df = tmp_df.loc[tmp_df['ΤΕΧΝΟΛΟΓΙΑ'] == tech]
                tmp2_df = tmp2_df[['ΤΕΧΝΟΛΟΓΙΑ']].groupby('ΤΕΧΝΟΛΟΓΙΑ').size().to_frame('ΣΥΝΟΛΟ').reset_index()
                if len(tmp2_df['ΣΥΝΟΛΟ'] > 0):
                    tech_values[tech].append(tmp2_df['ΣΥΝΟΛΟ'][0])
                else:
                    tech_values[tech].append(0)

        for tech in technologies:
            st_df[tech] = tech_values[tech]

        st_df = st_df.rename(columns={'ΠΕΡΙΦΕΡΕΙΑ': 'index'}).set_index('index')
        st.table(st_df)
        st_df.drop('ΣΥΝΟΛΟ', axis=1, inplace=True)
        st.bar_chart(st_df)


def visualize_data_per_technology(df: pd.DataFrame, attribute_type_options, tech_options):
    regions = df['ΠΕΡΙΦΕΡΕΙΑ'].unique()
    reg_values = {}
    for reg in regions:
        reg_values[reg] = []

    df = df[df['ΤΕΧΝΟΛΟΓΙΑ'].isin(tech_options)]
    tech_options.sort()

    if 'Μέγιστη Ισχύς (MW)' in attribute_type_options:
        st_df = df[['ΤΕΧΝΟΛΟΓΙΑ', 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby('ΤΕΧΝΟΛΟΓΙΑ').sum().reset_index()
        for tech in tech_options:
            if tech not in st_df['ΤΕΧΝΟΛΟΓΙΑ'].tolist():
                st_df.loc[-1] = [tech, 0]

        st_df = st_df.sort_values('ΤΕΧΝΟΛΟΓΙΑ')

        for tech in tech_options:
            tmp_df = df.loc[df['ΤΕΧΝΟΛΟΓΙΑ'] == tech]
            for reg in regions:
                tmp2_df = tmp_df.loc[tmp_df['ΠΕΡΙΦΕΡΕΙΑ'] == reg]
                tmp2_df = tmp2_df[['ΠΕΡΙΦΕΡΕΙΑ', 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby('ΠΕΡΙΦΕΡΕΙΑ').sum().reset_index()
                if len(tmp2_df['ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)'] > 0):
                    reg_values[reg].append(tmp2_df['ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)'][0])
                else:
                    reg_values[reg].append(0)

        for reg in regions:
            st_df[reg] = reg_values[reg]

        st_df = st_df.rename(columns={'ΤΕΧΝΟΛΟΓΙΑ': 'index'}).set_index('index')
        st.table(st_df)
        st_df.drop('ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)', axis=1, inplace=True)
        st.bar_chart(st_df)
    elif 'Αριθμός Α.Π.Ε' in attribute_type_options:
        st_df = df[['ΤΕΧΝΟΛΟΓΙΑ']].groupby('ΤΕΧΝΟΛΟΓΙΑ').size().to_frame('ΣΥΝΟΛΟ').reset_index()
        for tech in tech_options:
            if tech not in st_df['ΤΕΧΝΟΛΟΓΙΑ'].tolist():
                st_df.loc[-1] = [tech, 0]

        st_df = st_df.sort_values('ΤΕΧΝΟΛΟΓΙΑ')

        for tech in tech_options:
            tmp_df = df.loc[df['ΤΕΧΝΟΛΟΓΙΑ'] == tech]
            for reg in regions:
                tmp2_df = tmp_df.loc[tmp_df['ΠΕΡΙΦΕΡΕΙΑ'] == reg]
                tmp2_df = tmp2_df[['ΠΕΡΙΦΕΡΕΙΑ']].groupby('ΠΕΡΙΦΕΡΕΙΑ').size().to_frame('ΣΥΝΟΛΟ').reset_index()
                if len(tmp2_df['ΣΥΝΟΛΟ'] > 0):
                    reg_values[reg].append(tmp2_df['ΣΥΝΟΛΟ'][0])
                else:
                    reg_values[reg].append(0)

        for reg in regions:
            st_df[reg] = reg_values[reg]

        st_df = st_df.rename(columns={'ΤΕΧΝΟΛΟΓΙΑ': 'index'}).set_index('index')
        st.table(st_df)
        st_df.drop('ΣΥΝΟΛΟ', axis=1, inplace=True)
        st.bar_chart(st_df)


def visualize_duration_per_column(df: pd.DataFrame, column, period_options):
    st_df = pd.DataFrame()
    if period_options == 'Διάρκεια Α.Π.Ε.':
        st_df = df[[column, 'ΔΙΑΡΚΕΙΑ ΑΔΕΙΑΣ ΣΕ ΜΗΝΕΣ']].groupby(column).mean().reset_index()
        st_df['ΔΙΑΡΚΕΙΑ ΑΔΕΙΑΣ ΣΕ ΜΗΝΕΣ'] = st_df['ΔΙΑΡΚΕΙΑ ΑΔΕΙΑΣ ΣΕ ΜΗΝΕΣ'].astype(int)
    elif period_options == 'Διάστημα Έγκρισης Α.Π.Ε.':
        st_df = df[[column, 'ΔΙΑΣΤΗΜΑ ΕΓΚΡΙΣΗΣ ΣΕ ΜΗΝΕΣ']].groupby(column).mean().reset_index()
        st_df['ΔΙΑΣΤΗΜΑ ΕΓΚΡΙΣΗΣ ΣΕ ΜΗΝΕΣ'] = st_df['ΔΙΑΣΤΗΜΑ ΕΓΚΡΙΣΗΣ ΣΕ ΜΗΝΕΣ'].astype(int)

    st.table(st_df)
    st_df = st_df.rename(columns={column: 'index'}).set_index('index')
    st.bar_chart(st_df)


def visualize_regions_in_time(df: pd.DataFrame, period_options, attribute_type_options, region_options):
    column = ''
    if period_options == 'Μήνας':
        column = 'ΜΗΝΑΣ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'
    elif period_options == 'Εβδομάδα':
        column = 'ΕΒΔΟΜΑΔΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'

    unique_dates = df[column].unique()
    unique_dates.sort()

    st_df = pd.DataFrame()
    st_df[column] = unique_dates
    for region in region_options:
        st_df[region] = [0 for i in range(len(unique_dates))]

    for region in region_options:
        tmp_df = df.loc[df['ΠΕΡΙΦΕΡΕΙΑ'] == region]
        if 'Μέγιστη Ισχύς (MW)' in attribute_type_options:
            tmp_df = tmp_df[[column, 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby(column).sum().reset_index()
            tmp_df = tmp_df.rename(columns={'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)': 'result'})
        elif 'Αριθμός Α.Π.Ε' in attribute_type_options:
            tmp_df = tmp_df.groupby(column).size().to_frame('result').reset_index()

        result = []
        for date in unique_dates:
            row = tmp_df.loc[tmp_df[column] == date]
            if row.empty:
                result.append(0)
            else:
                result.append(row.iloc[0]['result'])

        st_df[region] = result

    st_df = st_df.rename(columns={column: 'index'}).set_index('index')
    st.line_chart(st_df)


def visualize_technology_in_time(df: pd.DataFrame, period_options, attribute_type_options, tech_options):
    column = ''
    if period_options == 'Μήνας':
        column = 'ΜΗΝΑΣ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'
    elif period_options == 'Εβδομάδα':
        column = 'ΕΒΔΟΜΑΔΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'

    unique_dates = df[column].unique()
    unique_dates.sort()

    st_df = pd.DataFrame()
    st_df[column] = unique_dates
    for tech in tech_options:
        st_df[tech] = [0 for i in range(len(unique_dates))]

    for tech in tech_options:
        tmp_df = df.loc[df['ΤΕΧΝΟΛΟΓΙΑ'] == tech]
        if 'Μέγιστη Ισχύς (MW)' in attribute_type_options:
            tmp_df = tmp_df[[column, 'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)']].groupby(column).sum().reset_index()
            tmp_df = tmp_df.rename(columns={'ΜΕΓΙΣΤΗ ΙΣΧΥΣ (MW)': 'result'})
        elif 'Αριθμός Α.Π.Ε' in attribute_type_options:
            tmp_df = tmp_df.groupby(column).size().to_frame('result').reset_index()

        result = []
        for date in unique_dates:
            row = tmp_df.loc[tmp_df[column] == date]
            if row.empty:
                result.append(0)
            else:
                result.append(row.iloc[0]['result'])

        st_df[tech] = result

    st_df = st_df.rename(columns={column: 'index'}).set_index('index')
    st.line_chart(st_df)


def visualize_wind_speeds_per_region(df: pd.DataFrame, region_options):
    df = df[df['ΠΕΡΙΦΕΡΕΙΑ'].isin(region_options)]
    st_df = df[['ΠΕΡΙΦΕΡΕΙΑ', 'ΠΕΡΙΦΕΡΕΙΑΚΗ ΜΕΣΗ ΤΑΧΥΤΗΤΑ ΑΕΡΑ Η80', 'ΠΕΡΙΦΕΡΕΙΑΚΗ ΜΕΣΗ ΤΑΧΥΤΗΤΑ ΑΕΡΑ Η100', 'ΠΕΡΙΦΕΡΕΙΑΚΗ ΜΕΣΗ ΤΑΧΥΤΗΤΑ ΑΕΡΑ Η120']].groupby('ΠΕΡΙΦΕΡΕΙΑ').mean().reset_index()

    st.table(st_df)

    fig = px.histogram(st_df,
                       x='ΠΕΡΙΦΕΡΕΙΑ',
                       y=['ΠΕΡΙΦΕΡΕΙΑΚΗ ΜΕΣΗ ΤΑΧΥΤΗΤΑ ΑΕΡΑ Η80', 'ΠΕΡΙΦΕΡΕΙΑΚΗ ΜΕΣΗ ΤΑΧΥΤΗΤΑ ΑΕΡΑ Η100', 'ΠΕΡΙΦΕΡΕΙΑΚΗ ΜΕΣΗ ΤΑΧΥΤΗΤΑ ΑΕΡΑ Η120'],
                       barmode='group')

    fig.update_layout(legend=dict(
        orientation="h",
        entrywidth=250,
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    st.plotly_chart(fig, use_container_width=True)


def main():
    pd.set_option("display.max.columns", None)

    st.set_page_config(layout="wide")

    df = load_df()

    # Streamlit
    st.title('Άδειες Α.Π.Ε. - Ανάλυση & Πρόβλεψη')
    st.sidebar.title('Mini Project ΙI: Data Visualization & Forecasting')
    st.sidebar.markdown('_By **Filitsa Ioanna Kouskouveli** and ***Vasilis Andritsoudis***_')
    st.sidebar.divider()

    # Create tabs
    tab_eda, tab_ml, tab_api = st.tabs(
        ['Ανάλυση Δεδομένων', 'Ανάλυση Forecasting Μοντέλων', 'Forecast (API)'])

    df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'] = pd.to_datetime(df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ']).dt.date

    date_options = st.sidebar.date_input(
        'Επιλογή Περιόδου',
        (df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'].min(), df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'].max()),
        df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'].min(),
        df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'].max(),
        format="DD/MM/YYYY"
    )

    period_options = st.sidebar.selectbox(
        'Επιλογή Χρονικού Διαστήματος',
        ('Μήνας', 'Εβδομάδα')
    )

    attribute_type_options = st.sidebar.selectbox(
        'Επιλογή Χαρακτηριστικού',
        ('Μέγιστη Ισχύς (MW)', 'Αριθμός Α.Π.Ε'),
    )

    timeseries_type_options = st.sidebar.multiselect(
        'Επιλογή Χρονοσειράς',
        ['Α.Π.Ε.', 'Τιμή CO2'],
        ['Α.Π.Ε.']
    )

    region_options = st.sidebar.multiselect(
        'Επιλογή Περιοχής',
        df['ΠΕΡΙΦΕΡΕΙΑ'].unique(),
        ['ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ', 'ΑΤΤΙΚΗΣ', 'ΘΡΑΚΗΣ', 'ΠΕΛΟΠΟΝΝΗΣΟΥ']
    )

    tech_options = st.sidebar.multiselect(
        'Επιλογή Είδος Α.Π.Ε.',
        df['ΤΕΧΝΟΛΟΓΙΑ'].unique(),
        ['ΦΩΤΟΒΟΛΤΑΪΚΑ', 'ΑΙΟΛΙΚΑ', 'ΜΥΗΕ']
    )

    period_type_options = st.sidebar.selectbox(
        'Επιλογή Τύπου Χρονικού Διαστήματος',
        ('Διάρκεια Α.Π.Ε.', 'Διάστημα Έγκρισης Α.Π.Ε.'),
    )

    if len(date_options) != 2:
        st.warning("Προσοχή: Παρακαλώ επιλέξτε περιόδο", icon="⚠️")
        return

    if len(timeseries_type_options) == 0:
        st.warning("Προσοχή: Παρακαλώ επιλέξτε χρονοσειρά", icon="⚠️")
        return

    if len(region_options) == 0:
        st.warning("Προσοχή: Παρακαλώ επιλέξτε περιοχή", icon="⚠️")
        return

    if len(tech_options) == 0:
        st.warning("Προσοχή: Παρακαλώ επιλέξτε είδος Α.Π.Ε.", icon="⚠️")
        return

    with tab_eda:
        with st.spinner("Φορτώνει..."):
            date_mask = (df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'] > date_options[0]) & (
                    df['ΗΜΕΡΟΜΗΝΙΑ ΕΚΔ. ΑΔ.ΠΑΡΑΓΩΓΗΣ'] <= date_options[1])
            df = df.loc[date_mask]

            st.subheader('Άδειες Α.Π.Ε. στον Χρόνο')
            visualize_timeseries(df, period_options, attribute_type_options, timeseries_type_options)

            st.divider()
            st.subheader('Άδειες Α.Π.Ε. στον Χρόνο ανά Περιοχή')
            visualize_regions_in_time(df, period_options, attribute_type_options, region_options)

            st.divider()
            st.subheader('Άδειες Α.Π.Ε. στον Χρόνο ανά Είδος')
            visualize_technology_in_time(df, period_options, attribute_type_options, tech_options)

            st.divider()
            st.subheader('Άδειες Α.Π.Ε. ανά Περιοχή')
            visualize_data_per_region(df, attribute_type_options, region_options)

            st.divider()
            st.subheader('Άδειες Α.Π.Ε. ανά Είδος')
            visualize_data_per_technology(df, attribute_type_options, tech_options)

            st.divider()
            st.subheader('Μέση Διάρκεια/Έγκριση Α.Π.Ε. ανά Περιοχή')
            visualize_duration_per_column(df, 'ΠΕΡΙΦΕΡΕΙΑ', period_type_options)

            st.divider()
            st.subheader('Μέση Διάρκεια/Έγκριση Α.Π.Ε. ανά Είδος Α.Π.Ε.')
            visualize_duration_per_column(df, 'ΤΕΧΝΟΛΟΓΙΑ', period_type_options)

            visualize_wind_speeds_per_region(df, region_options)

    with tab_ml:
        with st.spinner("Φορτώνει..."):
            df = pd.read_csv('results\\final_permits.csv')
            df = fm.preprocess_data(df)

            st.title("Ανάλυση Forecasting Μοντέλων")

            if period_options == 'Εβδομάδα':
                cols_to_drop_list = ['start_production_month', 'approval_period_in_months', 'approved_time_in_months',
                                     'monthly_co2_price']
                time_col = 'start_production_week'
                time_period = 'W'
            else:
                cols_to_drop_list = ['start_production_week', 'approval_period_in_weeks', 'approved_time_in_weeks',
                                     'weekly_co2_price']
                time_col = 'start_production_month'
                time_period = 'M'

            if attribute_type_options == 'Μέγιστη Ισχύς (MW)':
                target_var = 'total_mw'
            else:
                target_var = 'num_permits'

            k = st.slider('Επιλογή περιόδου πρόβλεψης', 1, 50, 20)
            lags = st.slider('Επιλογή διαστήματος καθυστέρησης (Lag)', 1, 20, 4)

            df_aggr = fm.transform_data_on_time_level(df, cols_to_drop_list, time_col, time_period)

            if st.button('Πρόβλεψη'):
                fm.predict_with_dif_models(df_aggr, k, target_var, time_col, time_period, lags)

    with tab_api:
        with st.spinner("Φορτώνει..."):
            df = pd.read_csv('results\\final_permits.csv')
            df = fm.preprocess_data(df)

            st.title("Forecast (API)")

            model_options = st.selectbox(
                'Επιλογή Μοντέλου',
                ('Sarima', 'Prophet'),
                key='model_api'
            )

            k = st.slider('Επιλογή περιόδου πρόβλεψης', 1, 50, 20, key='k_api')

            model = 'prophet'
            if model_options == 'Sarima':
                model = 'sarima'

            period = 'month'
            time_col = 'start_production_month'
            freq = 'M'
            if period_options == 'Εβδομάδα':
                period = 'week'
                time_col = 'start_production_week'
                freq = 'W'

            attribute = 'num_permits'
            if attribute_type_options == 'Μέγιστη Ισχύς (MW)':
                attribute = 'total_mw'

            df_aggr = fm.transform_data_on_time_level(df, cols_to_drop_list, time_col, time_period)

            if st.button('Πρόβλεψη', key='predict_api'):
                url = 'http://127.0.0.1:5000/' + model + '/' + period + '/' + attribute
                params = {'future': k}

                response = requests.get(url, params=params)

                predictions = pd.DataFrame(response.json())

                fm.plot_forecast(df_aggr, predictions, df_aggr, k, attribute, time_col, freq)


if __name__ == '__main__':
    main()
