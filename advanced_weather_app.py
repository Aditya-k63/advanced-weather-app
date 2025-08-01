import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import joblib

@st.cache_data
def load_data(file_path="DailyDelhiClimateTrain.csv"):
   
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' not found. Please upload it using the file uploader below.")
        return None
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        # Convert the date column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return None

def add_features(df):
   
    if 'date' not in df.columns or df['date'].isnull().all():
        raise ValueError("Date column is missing or empty.")
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lag and rolling features
    df['temp_diff'] = df['meantemp'].diff().fillna(0)
    df['rolling_mean_3'] = df['meantemp'].rolling(window=3).mean().fillna(df['meantemp'])

    # Categorical features
    df['temp_category'] = pd.cut(df['meantemp'], bins=[-float('inf'), 15, 25, float('inf')], labels=['Cold', 'Warm', 'Hot'])
    
    return df

def train_model(X, y, model_type='Linear Regression'):
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=42)
    else:
        
        model = LinearRegression()
     
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
  
    return {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False)
    }

def forecast_with_prophet(df):
    
    prophet_df = df[['date', 'meantemp']].rename(columns={"date": "ds", "meantemp": "y"})
    
    m = Prophet()
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def main():
    """
    The main function that sets up the Streamlit application.
    """
    
    st.title("Delhi Climate Analyzer and Temperature Predictor")

    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload 'DailyDelhiClimateTrain.csv'", type=['csv'])
    
    df = None  
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.dropna(inplace=True)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception as e:
            st.error(f"Error loading the uploaded file: {e}")
            
    if df is None: 
        df = load_data()

    if df is not None:
        
 
        df = add_features(df)

        with st.sidebar:
            st.header("Model Configuration")
            
            model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])
            
            selected_features = st.multiselect(
                "Select Features", 
                df.columns.difference(['date', 'meantemp', 'temp_category']), 
                default=['humidity', 'wind_speed', 'meanpressure', 'month', 'dayofweek', 'is_weekend', 'temp_diff']
            )
          
            st.header("Display Options")
            show_data = st.checkbox("Show Raw Data")
            show_chart = st.checkbox("Show Charts")
            show_summary = st.checkbox("Show Summary")
            show_forecast = st.checkbox("Show Forecast")
            enable_download = st.checkbox("Enable Prediction Download")

        if show_data:
            st.subheader("Raw Data")
            st.write(df.head())

        if show_chart:
            st.subheader("Temperature Trend")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['date'], df['meantemp'], color='tab:blue')
            ax.set_title("Daily Mean Temperature")
            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature (°C)")
            st.pyplot(fig)
            st.subheader("Correlation Heatmap")
            fig2, ax2 = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)

        if show_summary:
            st.subheader("Temperature Categories")
            st.write(df['temp_category'].value_counts())
            st.bar_chart(df['temp_category'].value_counts())

        st.subheader("Predict Mean Temperature")
        
        X = df[selected_features]
        y = df['meantemp']
        model, X_test, y_test, y_pred = train_model(X, y, model_type=model_choice)
        metrics = evaluate_model(y_test, y_pred)

        st.write("Model Performance:")
        st.json(metrics)
        st.write("Actual vs Predicted")
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.line_chart(results_df.reset_index(drop=True))
        if enable_download:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions as CSV", 
                data=csv, 
                file_name="predictions.csv", 
                mime='text/csv'
            )
        st.subheader("Try Predicting")
        user_inputs = []
        for col in selected_features:
            val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            user_inputs.append(val)
        user_input_df = pd.DataFrame([user_inputs], columns=selected_features)
        prediction = model.predict(user_input_df)[0]
        st.success(f"Predicted Temperature: {prediction:.2f} °C")
        if show_forecast:
            st.subheader("30-Day Temperature Forecast")
            forecast_df = forecast_with_prophet(df)
            st.line_chart(forecast_df.set_index("ds")["yhat"])

if __name__ == "__main__":
    main()
