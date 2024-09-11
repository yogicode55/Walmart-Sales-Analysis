import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest
import base64

df = pd.read_csv("Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df.dropna(subset=['Date'], inplace=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Day'] = df['Date'].dt.day

st.title("Walmart Sales Analysis")
st.sidebar.header("Filter by Store")

store_options = df['Store'].unique()
selected_store = st.sidebar.selectbox("Select Store", store_options)
store_data = df[df['Store'] == selected_store]

st.subheader(f"Sales Summary for Store {selected_store}")
st.write(store_data.describe())

# Weekly, Monthly, Yearly Sales Trends
weekly_sales = store_data.groupby(['Date'])['Weekly_Sales'].sum().reset_index()
weekly_sales = weekly_sales.set_index('Date').asfreq('W')

monthly_sales = store_data.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
monthly_sales['Month_Year'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str)

yearly_sales = store_data.groupby(['Year'])['Weekly_Sales'].sum().reset_index()

st.subheader(f"Weekly Sales Trends for Store {selected_store}")
st.line_chart(weekly_sales)

st.subheader(f"Monthly Sales Trends for Store {selected_store}")
st.line_chart(monthly_sales.set_index('Month_Year')['Weekly_Sales'])

st.subheader(f"Yearly Sales Trends for Store {selected_store}")
st.line_chart(yearly_sales.set_index('Year')['Weekly_Sales'])

# ARIMA Model for Forecast
st.subheader(f"ARIMA Sales Forecast for Store {selected_store}")
forecast_period = st.slider("Select forecast period (weeks):", min_value=4, max_value=52, value=12)

auto_model = auto_arima(weekly_sales['Weekly_Sales'], seasonal=False, stepwise=True)
model_fit = auto_model.fit(weekly_sales['Weekly_Sales'])
forecast = model_fit.predict(n_periods=forecast_period)

forecast_dates = pd.date_range(start=weekly_sales.index.max(), periods=forecast_period + 1, freq='W')[1:]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Sales': forecast})
st.line_chart(pd.concat([weekly_sales, forecast_df.set_index('Date')], axis=1))

# Sales Distribution
st.subheader(f"Sales Distribution for Store {selected_store}")
fig, ax = plt.subplots(figsize=(10, 6))  
sns.histplot(store_data['Weekly_Sales'], kde=True, ax=ax)
st.pyplot(fig)

# Correlation Matrix
st.subheader(f"Correlation Matrix for Store {selected_store}")
correlation_matrix = store_data.corr()
fig, ax = plt.subplots(figsize=(10, 8))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Store Locations Visualization
st.subheader("Store Locations and Sales Visualization")
store_locations = df[['Store', 'Date', 'Weekly_Sales']].groupby('Store').agg({'Weekly_Sales': 'sum'}).reset_index()
store_locations['Lat'] = np.random.uniform(low=25.0, high=50.0, size=len(store_locations))
store_locations['Lon'] = np.random.uniform(low=-125.0, high=-70.0, size=len(store_locations))

m = folium.Map(location=[37, -102], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)
for idx, row in store_locations.iterrows():
    folium.Marker([row['Lat'], row['Lon']],
                  popup=f"Store: {row['Store']}<br>Sales: ${row['Weekly_Sales']:.2f}").add_to(marker_cluster)
st_folium(m, width=700)

# Anomaly Detection
st.subheader(f"Anomaly Detection for Store {selected_store}")
iso_forest = IsolationForest(contamination=0.05)
weekly_sales['Anomaly'] = iso_forest.fit_predict(weekly_sales[['Weekly_Sales']])
anomalies = weekly_sales[weekly_sales['Anomaly'] == -1]
st.write("Detected anomalies in sales:")
st.write(anomalies)

fig, ax = plt.subplots(figsize=(14, 7))  
ax.plot(weekly_sales.index, weekly_sales['Weekly_Sales'], label='Sales')
ax.scatter(anomalies.index, anomalies['Weekly_Sales'], color='red', label='Anomalies')
ax.set_title(f"Sales Anomalies for Store {selected_store}")
ax.set_xlabel('Date')
ax.set_ylabel('Weekly Sales')
ax.legend()
st.pyplot(fig)

# CSV Report Download
st.sidebar.header("Customizable Reports")
if st.sidebar.button("Generate Sales Report"):
    csv = store_data.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sales_report_store_{selected_store}.csv">Download Sales Report</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
