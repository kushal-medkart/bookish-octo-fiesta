
import pandas as pd
from prophet import Prophet
import numpy as np
import sys, logging, json

# Product Name
# Customers who bought that product
if len(sys.argv) > 1:
    productcode = int(sys.argv[1])
else:
    print('please, supply product code')
    exit()

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").disabled = True

lister = []
files = ['Oct_Detail.csv','Nov_Detail.csv', 'Dec_Detail.csv', 'Jan_Detail.csv', 'Feb_Detail.csv', 'Mar_Detail.csv']
selected_cols = ['RegionName','StoreName', 'BillDate', 'BillTime', 'BillNumber', 'CustomerCode', 'CustomerName', 'ProductCode', 'Quantity', 'BillSeries', 'ProductFullName']

for f_name in files:
    df = pd.read_csv('store-data/' + f_name, usecols=selected_cols)
    lister.append(df[~df['BillSeries'].str.contains('SR')])
salesdetail_montly = pd.concat(lister, axis=0, ignore_index=True)

CustomerProductPurchase = {}

salesdetail_montly = salesdetail_montly[salesdetail_montly['ProductCode'] == productcode]
for group_name, group_data in salesdetail_montly.groupby(['RegionName', 'StoreName']):
    for group_customer, customer_data in group_data.groupby(['CustomerCode']):
        customers_data = []
        for billNumber, customer_purchases in customer_data.groupby('BillNumber'):
            if len(customer_purchases) > 1:
                for index in range(len(customer_purchases)-1,1):
                    customer_purchases[0]['Quantity'] = str(int(customer_purchases[0]['Quantity']) + int(customer_purchases[index]['Quantity']))
            customers_data.append(customer_purchases.iloc[0])
        CustomerProductPurchase[group_customer[0]] = pd.concat(customers_data, axis=1, ignore_index=True).transpose()


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

def FuncRunModel(func):
    result = []
    for customer_code in CustomerProductPurchase:
        for product_code, product_purchase in CustomerProductPurchase[customer_code].groupby('ProductCode'):
            # select product greater than 2
            if len(product_purchase) < 2:
                continue
            product_purchase['bill_datetime'] = pd.to_datetime(product_purchase['BillDate'] + " " + product_purchase['BillTime'], format='%d/%m/%Y %H:%M%p')
            # getting ready for model
            product_purchase.rename(columns={'bill_datetime': 'ds', 'Quantity': 'y'}, inplace=True)
            result.append(func(product_purchase))
    print(json.dumps(result))

def SeasonalityModel(product_purchase):
    # Fit Prophet model with weekly and yearly seasonality
# Initialize Prophet model and add multiple custom seasonalities
    model = Prophet()
    model.add_seasonality(name='weekly', period=7, fourier_order=3)  # Weekly seasonality (period=7 days)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Monthly seasonality (period=30.5 days on average)
    model.add_seasonality(name='two_monthly', period=60, fourier_order=3)  # Bi-monthly seasonality (period=60 days)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)  # Yearly seasonality (period=365.25 days on average)
    
    model.fit(product_purchase)

    # Make future date predictions
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)

    # Extract relevant columns from the forecast DataFrame
    forecasted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Extract the prediction for the next date (last row of the forecasted_values DataFrame)
    next_date_prediction = forecasted_values.iloc[-1]['ds']

    forecasted_data = {
        "customer_name" : str(product_purchase['CustomerName'][0]),
        "customer_code" : str(product_purchase['CustomerCode'][0]),
        "timestamp" : str(next_date_prediction),
        "count" : str(len(product_purchase)),
        "region_name": str(product_purchase['RegionName'][0]),
        "store_name": str(product_purchase['StoreName'][0]),
        "product_name": str(product_purchase['ProductFullName'][0])
    }
    return forecasted_data

FuncRunModel(SeasonalityModel)