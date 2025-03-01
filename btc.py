import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import asyncio
import websockets
import json
import plotly.graph_objects as go


async def get_binance_realtime_price(symbol):
    """Fetches real-time price from Binance WebSocket."""
    uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
    try:
        async with websockets.connect(uri) as websocket:
            message = await websocket.recv()
            data = json.loads(message)
            price = float(data['c'])
            return price
    except Exception as e:
        st.error(f"Binance WebSocket Error: {e}")
        return None




def monte_carlo_simulation(crypto_name, start_date, prediction_date, prediction_price, binance_symbol):
    """Performs Monte Carlo simulation and returns a DataFrame with results."""
    try:
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        data = yf.download(crypto_name, start=start_date, end=end_date)
        if data.empty:
            st.error("Error: No data found for the specified period.")
            return None

        close_column = 'Close' if 'Close' in data.columns else 'close'
        returns = data[close_column].pct_change().dropna()
        mean = returns.mean()
        std_dev = returns.std()

        simulations = 10000
        results_price = []
        days = (datetime.datetime.strptime(prediction_date, "%Y-%m-%d") - datetime.datetime.strptime(end_date, "%Y-%m-%d")).days

        start_price = asyncio.run(get_binance_realtime_price(binance_symbol))
        if start_price is None:
            st.error("Error: Could not fetch real-time starting price from Binance.")
            return None

        for _ in range(simulations):
            prices = [start_price]
            for _ in range(days):
                random_return = np.random.normal(mean, std_dev)
                prices.append(prices[-1] * (1 + random_return))
            results_price.append(float(prices[-1]))

        results_df = pd.DataFrame({'prices': results_price})
        return results_df

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

async def get_realtime_price(symbol, price_placeholder, chart_placeholder):
    """Fetches and displays real-time price and chart using Binance WebSocket."""
    uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
    prices = []
    times = []

    async with websockets.connect(uri) as websocket:
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                price = float(data['c'])
                timestamp = datetime.datetime.fromtimestamp(data['E'] / 1000)
                prices.append(price)
                times.append(timestamp)

                price_placeholder.markdown(f"<h1 style='text-align: center;'>Real-time Price ({symbol}): {price:.2f}</h1>", unsafe_allow_html=True) #make price big.

                fig = go.Figure(data=go.Scatter(x=times, y=prices, mode='lines'))
                fig.update_layout(title=f"Real-time Price Chart ({symbol})", xaxis_title="Time", yaxis_title="Price")
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                await asyncio.sleep(1)
            except websockets.exceptions.ConnectionClosed:
                st.error("WebSocket connection closed. Reconnecting...")
                await asyncio.sleep(5)
                return await get_realtime_price(symbol, price_placeholder, chart_placeholder)
            except Exception as e:
                st.error(f"WebSocket Error: {e}")
                await asyncio.sleep(5)
                return await get_realtime_price(symbol, price_placeholder, chart_placeholder)
            

def main():
    st.title("Crypto Price")

    tab1, tab2 = st.tabs(["Real-time Price", "Prediction"])

    with tab2:
        crypto_name = st.text_input("Crypto Name (e.g., BTC-USD):", "BTC-USD",key="crypto_name_input")
        binance_symbol = st.text_input("Binance Symbol (e.g., BTCUSDT):", "BTCUSDT", key="binance_symbol_input_prediction") #added binance symbol
        start_date = st.date_input("Start Date:", datetime.date(2025, 1, 1))
        prediction_date = st.date_input("Prediction Date:", datetime.date(2025, 3, 1))
        prediction_price = st.number_input("Prediction Price:")

        if st.button("Run Simulation"):
            results_df = monte_carlo_simulation(
                crypto_name,
                start_date.strftime("%Y-%m-%d"),
                prediction_date.strftime("%Y-%m-%d"),
                prediction_price,
                binance_symbol,
            )

            if results_df is not None:
                mean = results_df['prices'].mean()
                over_count = 0
                under_count = 0
                total_count = len(results_df)

                for price in results_df['prices']:
                    if price > prediction_price:
                        over_count += 1
                    elif price < prediction_price:
                        under_count += 1

                result = pd.DataFrame({
                    'Prediction Price': [mean, prediction_price],
                    'Over': ['50%', f'{round((over_count / total_count) * 100)}%'],
                    'Under': ['50%', f'{round((under_count / total_count) * 100)}%'],
                    'Over_odds': ['1.85', round(1/((over_count / total_count)*1.08),2)],
                    'Under_odds': ['1.85', round(1/((under_count / total_count)*1.08),2)]},
                    index=['Mean Simulated Price', 'User Prediction Price'])

                st.write(result)

    with tab1:
        symbol = st.text_input("Binance Symbol (e.g., BTCUSDT):", "BTCUSDT")
        price_placeholder = st.empty()
        chart_placeholder = st.empty()
        if st.button("Get real-time price:"):
            asyncio.run(get_realtime_price(symbol, price_placeholder, chart_placeholder))

if __name__ == "__main__":
    main()