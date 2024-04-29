
import pandas as pd
from prophet import Prophet
import numpy as np

lister = []
files = ['Oct_Detail.csv','Nov_Detail.csv', 'Dec_Detail.csv', 'Jan_Detail.csv', 'Feb_Detail.csv', 'Mar_Detail.csv']
selected_cols = ['RegionName','StoreName', 'BillDate', 'BillTime', 'BillNumber', 'CustomerCode', 'CustomerName', 'ProductCode', 'Quantity', 'BillSeries', 'ProductFullName']

for f_name in files:
    df = pd.read_csv('store-data/' + f_name, usecols=selected_cols)
    lister.append(df[~df['BillSeries'].str.contains('SR')])
salesdetail_montly = pd.concat(lister, axis=0, ignore_index=True)

CustomerProductPurchase = {}

salesdetail_montly = salesdetail_montly[salesdetail_montly['ProductCode'] == 265]
for group_name, group_data in salesdetail_montly.groupby(['RegionName', 'StoreName']):
    for group_customer, customer_data in group_data.groupby(['CustomerCode']):
        customers_data = []
        for billNumber, customer_purchases in customer_data.groupby('BillNumber'):
            for product_code, customer_purchase in customer_purchases.groupby(['ProductCode']):
              # same product is packed more than one time
                if len(customer_purchase) > 1:
                    for index in range(len(customer_purchase)-1,1):
                        customer_purchase[0]['Quantity'] = str(int(customer_purchase[0]['Quantity']) + int(customer_purchase[index]['Quantity']))

                customers_data.append(customer_purchase.iloc[0])
        CustomerProductPurchase[group_customer[0]] = pd.concat(customers_data, axis=1, ignore_index=True).transpose()


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

def FuncRunModel(func):
    for customer_code in CustomerProductPurchase:
        for product_code, product_purchase in CustomerProductPurchase[customer_code].groupby('ProductCode'):
            # select product greater than 2
            if len(product_purchase) < 2:
                continue
            product_purchase['bill_datetime'] = pd.to_datetime(product_purchase['BillDate'] + " " + product_purchase['BillTime'], format='%d/%m/%Y %H:%M%p')
            # getting ready for model
            product_purchase.rename(columns={'bill_datetime': 'ds', 'Quantity': 'y'}, inplace=True)
            func(product_purchase)

def SeasonalityModel(product_purchase):

    # Fit Prophet model with weekly and yearly seasonality
    model = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.add_seasonality(name='monthly', period=30, fourier_order=5)
    model.add_seasonality(name='twomontly', period=60, fourier_order=6)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    model.fit(product_purchase)

    # Make future date predictions
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)

    # Extract relevant columns from the forecast DataFrame
    forecasted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Extract the prediction for the next date (last row of the forecasted_values DataFrame)
    next_date_prediction = forecasted_values.iloc[-1]['ds']

    print(f"Predicted Next Date: {next_date_prediction}")
"""
    # Plot actual vs. predicted sales
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(product_purchase['ds'], product_purchase['y'], label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Actual vs. Forecasted Sales with Seasonality')
    ax.legend()
    plt.show()
"""