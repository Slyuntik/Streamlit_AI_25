import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
    return train, test

@st.cache_resource 
def load_model():
    with open('car_price_model.pickle', 'rb') as f:
        return pickle.load(f)

df_train, df_test = load_data()
df_eda = pd.concat([df_train, df_test], ignore_index=True)
artifacts = load_model()

model = artifacts['model']
encoder = artifacts['encoder']
feature_names = artifacts['feature_names']
metrics = artifacts['metrics']

def preprocess_input(df_input):
    df = df_input.copy()
    
    change_type_cols = ['mileage', 'engine', 'max_power']
    for col in change_type_cols:
        df[col] = df[col].str.split().str[0].astype('float')

    df['torque_value'] = df['torque'].str.split('@').str[0].str.lower().str.split('nm').str[0].str.split('kgm').str[0].str.split('/').str[0].str.strip().str.split('(').str[0].astype('float')
    df.loc[df['torque'].str.contains('kgm', na=False), 'torque_value'] *= 9.80665
    df['max_torque_rpm'] = df['torque'].str.lower().str.replace('@', '').str.replace('nm', '').str.replace(',', '').str.replace('/', '').str.split('rpm').str[0].str.replace('(kgm ', '').str.split().str[-1].str.split('-').str[-1].str.split('~').str[-1].str.replace('(', '').astype('float')
    df = df.drop('torque', axis=1).rename(columns={'torque_value': 'torque'})

    medians = {
        'year': 2014.00,
        'km_driven': 70000.00,
        'mileage': 19.37,
        'engine': 1248.00,
        'max_power': 81.86,
        'seats': 5.00,
        'torque': 160.00,
        'max_torque_rpm': 3000.00
    }

    df.fillna(medians, inplace=True)
    df[['engine', 'seats']] = df[['engine', 'seats']].astype('int')

    cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    df['name'] = df['name'].str.split().str[0]
    
    df[encoder.get_feature_names_out()] = encoder.transform(df[cat_cols])

    df['max_power_per_engine'] = df['max_power'] / df['engine']
    df['max_power_per_torque'] = df['max_power'] / df['torque']
    df['year_squared'] = df['year'] ** 2
    df['km_driven_log'] = np.log(df['km_driven'])
    df['owner_3_or_morе'] = df['owner'].apply(lambda x: 0 if (x.split()[0] == 'First' or x.split()[0] == 'Second') else 1)
    df = df.drop(cat_cols, axis=1)

    cols = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound
    
    return df[feature_names]

st.sidebar.markdown("# Car Price Prediction\n---")
page = st.sidebar.radio("", ["EDA", "CSV Prediction", "Form Prediction", "Model Information"])

if page == "EDA":
    st.markdown("## EDA")
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего записей", len(df_eda))
    col2.metric("Обучающая", len(df_train))
    col3.metric("Тестовая", len(df_test))
    
    col1, col2 = st.columns(2)
    col1.metric("Дубликаты", df_eda.duplicated().sum())
    col2.metric("Пропуски", df_eda.isnull().sum().sum())
    
    tab1, tab2, tab3, tab4 = st.tabs(["Обзор", "Числовые", "Категориальные", "Корреляции"])
    
    with tab1:
        st.dataframe(df_eda.head())
        buffer = StringIO()
        df_eda.info(buf=buffer)
        st.text(buffer.getvalue())
        st.write("Числовые признаки:")
        st.dataframe(df_eda.select_dtypes(include=[np.number]).describe())
        st.write("Категориальные признаки:")
        st.dataframe(df_eda.select_dtypes(include=['object']).describe())
    
    with tab2:
        num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        num_cols = [c for c in num_cols if c in df_eda.columns]
        col = st.selectbox("Признак:", num_cols)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        data = pd.to_numeric(df_eda[col], errors='coerce').dropna()
        if len(data) > 0:
            ax1.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax2.boxplot(data)
        else:
            ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Частота')
        ax2.set_ylabel(col)
        st.pyplot(fig)
    
    with tab3:
        cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']
        cat_cols = [c for c in cat_cols if c in df_eda.columns]
        col = st.selectbox("Признак:", cat_cols, key='cat')
        fig = px.bar(x=df_eda[col].value_counts().index, y=df_eda[col].value_counts().values)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
        if 'selling_price' in df_eda.columns:
            cols_for_corr = numeric_cols + ['selling_price']
        else:
            cols_for_corr = numeric_cols
        if len(cols_for_corr) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            corr_matrix = df_eda[cols_for_corr].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax1)
            if 'selling_price' in df_eda.columns:
                ax2.hist(df_eda['selling_price'].dropna(), bins=50, edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Цена')
                ax2.set_ylabel('Количество')
            else:
                ax2.text(0.5, 0.5, 'Нет данных о цене', ha='center', va='center')
            st.pyplot(fig)

elif page == "CSV Prediction":
    st.markdown("## CSV Prediction")
    uploaded = st.file_uploader("Загрузите CSV", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Predict"):
            try:
                processed = preprocess_input(df)
                preds = model.predict(processed)
                df['predicted_price'] = preds.round().astype(int)
                st.write(f"Predictions: {len(preds)}")
                display_cols = []
                if 'name' in df.columns:
                    display_cols.append('name')
                display_cols.extend(['year', 'km_driven', 'fuel', 'predicted_price'])
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols].head())
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif page == "Form Prediction":
    st.markdown("## Form Prediction")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Maruti Swift VXI")
        year = st.slider("Year", 1990, 2025, 2018)
        km_driven = st.number_input("Km Driven", 0, 1000000, 50000)
        fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller_type = st.selectbox("Seller", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    with col2:
        owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        mileage = st.text_input("Mileage", "18.9 kmpl")
        engine = st.text_input("Engine", "1197 CC")
        max_power = st.text_input("Max Power", "82 bhp")
        torque = st.text_input("Torque", "114Nm@ 4000rpm")
        seats = st.selectbox("Seats", [2, 3, 4, 5, 6, 7, 8, 9, 10], index=3)
    if st.button("Predict Price"):
        data = pd.DataFrame({
            'name': [name], 'year': [year], 'km_driven': [km_driven], 'fuel': [fuel],
            'seller_type': [seller_type], 'transmission': [transmission], 'owner': [owner],
            'mileage': [mileage], 'engine': [engine], 'max_power': [max_power],
            'torque': [torque], 'seats': [seats]
        })
        try:
            processed = preprocess_input(data)
            pred = model.predict(processed)[0]
            st.write(f"Predicted Price: {int(round(pred))}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif page == "Model Information":
    st.markdown("## Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Model Type: {type(model).__name__}")
        st.write(f"Number of Features: {len(feature_names)}")
        st.write(f"Train MSE: {metrics['train_mse']:,.0f}")
        st.write(f"Train R²: {metrics['train_r2']:.4f}")
    with col2:
        st.write(f"Encoder Type: {type(encoder).__name__}")
        st.write(f"Test MSE: {metrics['test_mse']:,.0f}")
        st.write(f"Test R²: {metrics['test_r2']:.4f}")
    
    if hasattr(model, 'coef_'):
        st.markdown("### Model Weights (Top 20)")
        coefs = model.coef_
        if len(coefs) == len(feature_names):
            df_coef = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs,
                'Abs': np.abs(coefs)
            }).sort_values('Abs', ascending=False)
            fig = px.bar(df_coef.head(20), x='Coefficient', y='Feature', orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)