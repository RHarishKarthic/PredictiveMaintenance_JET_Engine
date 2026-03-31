
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. LOAD ARTIFACTS ---
# Use a try-except block to handle potential FileNotFoundError
try:
    model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    metadata = joblib.load('metadata.joblib')
except FileNotFoundError:
    st.error("Model files not found! Please run `build_model.py` first.")
    st.stop()

features = metadata['features']
remaining_sensors = metadata['remaining_sensors']

# --- App Config ---
st.set_page_config(page_title="Jet Engine RUL Predictor", layout="wide", initial_sidebar_state="expanded")

# --- 2. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    try:
        # Load FD001 data
        train_df = pd.read_csv('CMAPSSData/train_FD001.txt', sep=r'\s+', header=None, names=col_names)
        test_df = pd.read_csv('CMAPSSData/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
        y_test = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    except FileNotFoundError:
        st.error("Dataset files not found in the `CMAPSSData` folder. Please download and place them correctly.")
        st.stop()

    # Add RUL to train_df for degradation curve
    max_cycle = train_df.groupby('unit_nr')['time_cycles'].transform('max')
    train_df['RUL'] = max_cycle - train_df['time_cycles']
    
    return train_df, test_df, y_test

train_df, test_df, y_test = load_data()

# --- 3. FEATURE ENGINEERING & PREDICTION FUNCTIONS ---
def add_rolling_features(df, sensors, window=5):
    df_out = df.copy()
    for s in sensors:
        df_out[s + '_rolling_mean'] = df_out.groupby('unit_nr')[s].transform(lambda x: x.rolling(window).mean())
        df_out[s + '_rolling_std'] = df_out.groupby('unit_nr')[s].transform(lambda x: x.rolling(window).std())
    return df_out.fillna(0)

def make_rul_prediction(df, features_list, scaler_obj, model_obj):
    processed_df = add_rolling_features(df, remaining_sensors)
    # Ensure all feature columns exist, fill missing with 0
    for col in features_list:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[features_list] # Keep order
    
    processed_df = scaler_obj.transform(processed_df)
    predictions = model_obj.predict(processed_df)
    return predictions

# --- 4. UI LAYOUT ---
st.title("✈️ Predictive Maintenance: Jet Engine RUL Predictor")

# --- "About the Data" Expander ---
with st.expander("ℹ️ About the Dataset & Terminology"):
    st.markdown("""
    This dashboard uses the **NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)**.
    
    - **What is a 'Cycle'?** An operational cycle represents a single flight of the engine, from start to shutdown.
    
    - **What are 'Operational Settings'?** These are control inputs to the engine that affect its performance, such as altitude, speed, and temperature settings.
    
    - **What are 'Sensor Measurements'?** The dataset includes readings from 21 sensors measuring things like temperature, pressure, and fan speed. The model uses the patterns in these sensor readings to detect degradation.
    
    - **What is 'RUL'?** Remaining Useful Life is the number of operational cycles an engine has left before it is predicted to fail.
    """)

# --- Sidebar for Engine Selection ---
st.sidebar.header("Engine Selection")
engine_id = st.sidebar.selectbox("Select Engine Unit ID", test_df['unit_nr'].unique())

# --- Filter data for selected engine ---
engine_test_data = test_df[test_df['unit_nr'] == engine_id].copy()
engine_train_data = train_df[train_df['unit_nr'] == engine_id].copy()

# --- Make Predictions ---
final_prediction = make_rul_prediction(engine_test_data, features, scaler, model)
final_rul = final_prediction[-1]
actual_rul = y_test.iloc[int(engine_id) - 1]['RUL']

# --- Main Dashboard ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("RUL Prediction")
    
    # RUL Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=final_rul,
        title={'text': "Predicted RUL (Cycles)"},
        gauge={
            'axis': {'range': [0, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0.6)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#d9534f'},
                {'range': [50, 120], 'color': '#f0ad4e'},
                {'range': [120, 200], 'color': '#5cb85c'}
            ],
        },
        delta={'reference': actual_rul, 'increasing': {'color': "#333"}}
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.metric("Actual RUL (Ground Truth)", f"{int(actual_rul)} cycles")
    st.metric("Total Cycles Flown in Test Data", engine_test_data['time_cycles'].max())

with col2:
    st.subheader("Prediction Explanation", help="""
    This chart shows the most influential factors the model used to make its prediction. 
    A higher 'Importance Score' means the model paid more attention to that sensor's recent trend. 
    Features are often 'rolling means' or 'rolling stds', which represent the recent average and stability of a sensor's readings.
    """)
    
    # Feature Importance
    feature_imp = pd.DataFrame({
        'importance': model.feature_importances_, 
        'feature': features
    }).sort_values('importance', ascending=False).head(7)
    
    fig_imp = go.Figure(go.Bar(
        x=feature_imp['importance'],
        y=feature_imp['feature'],
        orientation='h',
        marker_color='#0275d8'
    ))
    fig_imp.update_layout(
        title="Top 7 Most Influential Factors",
        xaxis_title="Importance Score",
        yaxis_title="Sensor/Feature",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# --- Degradation and Sensor Analysis ---
st.subheader("Engine Degradation Analysis")
tab1, tab2 = st.tabs(["RUL Degradation Curve", "Sensor Health Trends"])

with tab1:
    st.subheader("RUL Degradation Curve", help="""
    This graph visualizes the engine's health over its entire lifespan.
    - **Actual RUL (Ground Truth):** This dotted black line shows the perfect, linear countdown to failure. It represents the true remaining life at any given cycle.
    - **Predicted RUL:** This solid red line is the model's prediction at each cycle. The model only uses sensor data to make its guess. A good model will have a prediction line that trends downwards and stays close to the actual RUL line, especially towards the end of the engine's life.
    """)
    
    # Predict RUL over the engine's life
    full_history_data = pd.concat([engine_train_data, engine_test_data]).sort_values('time_cycles')
    rul_over_time = make_rul_prediction(full_history_data, features, scaler, model)
    
    # Create the degradation plot
    fig_deg = go.Figure()
    fig_deg.add_trace(go.Scatter(
        x=full_history_data['time_cycles'], 
        y=full_history_data['RUL'], 
        mode='lines', 
        name='Actual RUL (Ground Truth)',
        line=dict(color='lightgreen', dash='dash', width=2)
    ))
    fig_deg.add_trace(go.Scatter(
        x=full_history_data['time_cycles'], 
        y=rul_over_time, 
        mode='lines', 
        name='Predicted RUL',
        line=dict(color='#d9534f', width=2.5)
    ))
    fig_deg.update_layout(
        title=f"RUL Degradation Curve for Engine {engine_id}",
        xaxis_title="Time (Cycles)",
        yaxis_title="Remaining Useful Life (RUL)",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
    )
    st.plotly_chart(fig_deg, use_container_width=True)

with tab2:
    col_sensor_1, col_sensor_2 = st.columns([1, 2])
    
    with col_sensor_1:
        sensor_to_plot = st.selectbox("Choose Sensor to Visualize", remaining_sensors, key="sensor_select")
        
        # Sensor Stats
        st.markdown(f"**Statistics for `{sensor_to_plot}`**")
        current_val = engine_test_data[sensor_to_plot].iloc[-1]
        avg_val = engine_test_data[sensor_to_plot].mean()
        max_val = engine_test_data[sensor_to_plot].max()
        
        st.metric(f"Current Value", f"{current_val:.2f}")
        st.metric(f"Average Value (in test data)", f"{avg_val:.2f}")
        st.metric(f"Max Value (in test data)", f"{max_val:.2f}")

    with col_sensor_2:
        fig_sensor = go.Figure()
        fig_sensor.add_trace(go.Scatter(
            x=engine_test_data['time_cycles'], 
            y=engine_test_data[sensor_to_plot], 
            mode='lines', 
            name='Raw Signal',
            line=dict(color='#0275d8', width=1.5)
        ))
        rolling_mean = engine_test_data[sensor_to_plot].rolling(window=10).mean()
        fig_sensor.add_trace(go.Scatter(
            x=engine_test_data['time_cycles'], 
            y=rolling_mean, 
            mode='lines', 
            name='Trend (10-Cycle Avg)',
            line=dict(color='#d9534f', width=3)
        ))
        fig_sensor.update_layout(
            title=f"Sensor `{sensor_to_plot}` Trend for Engine {engine_id}",
            xaxis_title="Time (Cycles)",
            yaxis_title="Sensor Value",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_sensor, use_container_width=True)

# Show raw data if checkbox is ticked
if st.sidebar.checkbox("Show Raw Sensor Data for Selected Engine"):
    st.sidebar.write(f"Displaying last 15 cycles for Engine {engine_id}")
    st.sidebar.dataframe(engine_test_data.tail(15))

st.markdown("---")
st.caption("Developed for ML Lab Project - Jet Engine Predictive Maintenance")