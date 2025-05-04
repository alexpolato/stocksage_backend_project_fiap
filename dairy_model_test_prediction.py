import pandas as pd
import numpy as np
import pickle
import datetime


def prepareDataforPrediction(formData):
    dados_path = r"C:\Users\alexa\OneDrive\Anexos\Fiap\stocksage_fiap_challenge\stocksage_backend\dairy\dairy_data\current_processed_data.csv"
    data = pd.read_csv(dados_path)

    current_date = datetime.datetime.now()
    production_date = datetime.datetime.strptime(
        formData["production_date"], "%Y-%m-%d"
    )
    expiration_date = datetime.datetime.strptime(
        formData["expiration_date"], "%Y-%m-%d"
    )

    product_name = formData["product_name"]
    quantity_before_sell = pd.to_numeric(
        formData["quantity_before_sell"], errors="coerce"
    )
    price_per_unit = pd.to_numeric(formData["price_per_unit"], errors="coerce")
    quantity_sold = pd.to_numeric(formData["quantity_sold"], errors="coerce")
    price_per_unit_sold = pd.to_numeric(
        formData["price_per_unit_sold"], errors="coerce"
    )

    shelf_life = (expiration_date - production_date).days
    days_to_expire = (expiration_date - current_date).days
    days_to_sell = (current_date - production_date).days
    quantity_in_stock = quantity_before_sell - quantity_sold
    sales_velocity = quantity_sold / days_to_sell

    # Obter valores históricos com base no produto
    product_data = data[data["Product Name"] == product_name]

    Quantity_Sold_Hist_Mean = product_data["Quantity_Sold_Hist_Mean"].min()
    Sales_Velocity_Hist_Mean = product_data["Sales_Velocity_Hist_Mean"].min()
    Sales_Volatility_Hist = product_data["Sales_Volatility_Hist"].min()
    Capacity_Utilization = quantity_sold / (Sales_Velocity_Hist_Mean * days_to_sell)
    Sales_Trend_7d = product_data["Sales_Trend_7d"].iloc[-1]
    Seasonal_Volatility = product_data["Seasonal_Volatility"].iloc[-1]

    stock_efficiency = (sales_velocity * shelf_life) / (quantity_in_stock + 1)
    total_value = quantity_in_stock * price_per_unit

    prediction_data = {
        "product_name": product_name,
        "price_per_unit": price_per_unit,
        "price_per_unit_sold": price_per_unit_sold,
        "shelf_life": shelf_life,
        "days_to_expire": days_to_expire,
        "days_to_sell": days_to_sell,
        "quantity_in_stock": quantity_in_stock,
        "sales_velocity": sales_velocity,
        "Quantity_Sold_Hist_Mean": Quantity_Sold_Hist_Mean,
        "Sales_Velocity_Hist_Mean": Sales_Velocity_Hist_Mean,
        "Sales_Volatility_Hist": Sales_Volatility_Hist,
        "Capacity_Utilization": Capacity_Utilization,
        "Sales_Trend_7d": Sales_Trend_7d,
        "Seasonal_Volatility": Seasonal_Volatility,
        "stock_efficiency": stock_efficiency,
        "total_value": total_value,
    }
    return prediction_data


def model_prediction(pred_data):

    # Carregar modelo e pré-processadores
    final_model = pd.read_pickle("dairy/dairy_models/final_model.pkl")
    with open(
        "dairy/dairy_models/preprocessor.pkl", "rb"
    ) as f:  # Substitua pelo seu pré-processador
        preprocessor = pickle.load(f)

    # feature_lists = pd.DataFrame(feature_lists)

    """
    ['Product Name', 'Quantity (liters/kg)', 'Price per Unit', 'Total Value',
    'Shelf Life (days)', 'Price per Unit (sold)', 'Days_to_Sell', 'Days_to_Expire',
    'Sales_Velocity', 'Quantity_Sold_Hist_Mean', 'Sales_Velocity_Hist_Mean',
    'Sales_Volatility_Hist', 'Capacity_Utilization', 'Sales_Trend_7d',
    'Seasonal_Volatility', 'Stock_Efficiency']
    """
    # Criar dicionário de dados de exemplo
    example_data = {
        "Product Name": [pred_data["product_name"]],
        "Quantity (liters/kg)": [pred_data["quantity_in_stock"]],
        "Price per Unit": [pred_data["price_per_unit"]],
        "Total Value": [pred_data["total_value"]],
        "Shelf Life (days)": [pred_data["shelf_life"]],
        "Price per Unit (sold)": [pred_data["price_per_unit_sold"]],
        "Days_to_Sell": [pred_data["days_to_sell"]],
        "Days_to_Expire": [pred_data["days_to_expire"]],
        "Sales_Velocity": [pred_data["sales_velocity"]],
        "Quantity_Sold_Hist_Mean": [pred_data["Quantity_Sold_Hist_Mean"]],
        "Sales_Velocity_Hist_Mean": [pred_data["Sales_Velocity_Hist_Mean"]],
        "Sales_Volatility_Hist": [pred_data["Sales_Volatility_Hist"]],
        "Capacity_Utilization": [pred_data["Capacity_Utilization"]],
        "Sales_Trend_7d": [pred_data["Sales_Trend_7d"]],
        "Seasonal_Volatility": [pred_data["Seasonal_Volatility"]],
        "Stock_Efficiency": [pred_data["stock_efficiency"]],
    }

    # Converter para DataFrame
    example_df = pd.DataFrame(example_data)
    print("Exemplo de dados para teste de predição:\n", example_df)
    print("Sales velocity hist \n", example_df["Sales_Velocity_Hist_Mean"])

    X_novo = pd.DataFrame(example_df)

    # Converter para DataFrame
    # Aplicar pré-processamento
    X_processed = preprocessor.transform(X_novo)

    # Fazer predição
    prediction = final_model.predict(X_processed)
    return prediction


formData = {
    "production_date": (datetime.datetime.now() - datetime.timedelta(days=10)).strftime(
        "%Y-%m-%d"
    ),
    "expiration_date": (
        datetime.datetime.now()
        - datetime.timedelta(days=10)
        + datetime.timedelta(days=65)
    ).strftime("%Y-%m-%d"),
    "product_name": "Milk",
    "quantity_before_sell": 800,
    "price_per_unit": 28.0,
    "quantity_sold": 400,
    "price_per_unit_sold": 29.0,
}

data_prepared = prepareDataforPrediction(formData)
predict = model_prediction(data_prepared)
