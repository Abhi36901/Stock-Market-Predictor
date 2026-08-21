import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# ==========================================
# NEWS API
# ==========================================

NEWS_API_KEY = os.getenv(
    "NEWS_API_KEY",
    "PASTE_YOUR_NEWSAPI_KEY_HERE"
)


# ==========================================
# PREDICT STOCK
# ==========================================

def predict_stock(ticker):

    # ==========================================
    # 1. DOWNLOAD STOCK DATA
    # ==========================================

    data = yf.download(
        ticker,
        period="5y",
        progress=False
    )

    # If ticker does not exist
    if data.empty:
        return None


    # ==========================================
    # FIX YAHOO FINANCE MULTIINDEX
    # ==========================================

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = (
            data.columns
            .get_level_values(0)
        )


    # ==========================================
    # 2. CREATE FEATURES
    # ==========================================

    data["Return"] = (
        data["Close"].pct_change()
    )


    data["MA10"] = (
        data["Close"]
        .rolling(10)
        .mean()
    )


    data["MA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )


    data["Volatility"] = (
        data["Return"]
        .rolling(20)
        .std()
    )


    data["Momentum"] = (
        data["Close"]
        .pct_change(10)
    )


    # Next day's closing price
    data["Target"] = (
        data["Close"]
        .shift(-1)
    )


    # ==========================================
    # DIRECTION
    # ==========================================

    data["Direction"] = (
        data["Target"]
        > data["Close"]
    ).astype(int)


    # ==========================================
    # 3. FEATURES
    # ==========================================

    features = [

        "Close",

        "Return",

        "MA10",

        "MA50",

        "Volume",

        "Volatility",

        "Momentum"

    ]


    # ==========================================
    # 4. REMOVE INCOMPLETE TRAINING ROWS
    # ==========================================

    train_data = data.dropna(
        subset=features + ["Target"]
    ).copy()


    # Safety check
    if train_data.empty:
        return None


    X = train_data[features]

    y_price = train_data["Target"]

    y_direction = train_data["Direction"]


    # ==========================================
    # 5. PRICE MODEL
    # ==========================================

    price_model = LinearRegression()


    price_model.fit(
        X,
        y_price
    )


    # ==========================================
    # 6. DIRECTION MODEL
    # ==========================================

    direction_model = LogisticRegression(
        max_iter=1000
    )


    direction_model.fit(
        X,
        y_direction
    )


    # ==========================================
    # 7. LATEST DATA
    # ==========================================

    latest = data.iloc[-1]


    # Make sure latest features do not contain NaN
    latest_features = pd.DataFrame(
        [[
            latest["Close"],
            latest["Return"],
            latest["MA10"],
            latest["MA50"],
            latest["Volume"],
            latest["Volatility"],
            latest["Momentum"]
        ]],
        columns=features
    )


    if latest_features.isnull().values.any():

        return None


    # ==========================================
    # 8. CURRENT PRICE
    # ==========================================

    original_current_price = float(
        latest["Close"]
    )


    # ==========================================
    # 9. NEXT-DAY PRICE PREDICTION
    # ==========================================

    predicted_price = float(
        price_model.predict(
            latest_features
        )[0]
    )


    expected_change = (
        predicted_price
        - original_current_price
    )


    expected_change_percent = (
        expected_change
        / original_current_price
    ) * 100


    # ==========================================
    # 10. DIRECTION PREDICTION
    # ==========================================

    direction_prediction = (
        direction_model.predict(
            latest_features
        )[0]
    )


    probabilities = (
        direction_model.predict_proba(
            latest_features
        )[0]
    )


    if direction_prediction == 1:

        direction = "UP"

        confidence = probabilities[1]

    else:

        direction = "DOWN"

        confidence = probabilities[0]


    # ==========================================
    # 11. NEWS SENTIMENT
    # ==========================================

    analyzer = SentimentIntensityAnalyzer()

    sentiments = []

    latest_news = []


    try:

        url = (
            "https://newsapi.org/v2/everything"
        )


        params = {

            "q": f'"{ticker}"',

            "language": "en",

            "sortBy": "publishedAt",

            "pageSize": 10,

            "apiKey": NEWS_API_KEY

        }


        response = requests.get(
            url,
            params=params,
            timeout=10
        )


        news_data = response.json()


        if news_data.get("status") == "ok":

            for article in news_data.get(
                "articles",
                []
            ):

                title = (
                    article.get("title")
                    or ""
                ).strip()


                description = (
                    article.get("description")
                    or ""
                ).strip()


                # Ignore empty headlines
                if not title:
                    continue


                # Remove weird "[Removed]" articles
                if title.lower() == "[removed]":
                    continue


                text = (
                    title
                    + " "
                    + description
                )


                sentiment = (
                    analyzer
                    .polarity_scores(text)
                    ["compound"]
                )


                sentiments.append(
                    sentiment
                )


                latest_news.append({

                    "title": title,

                    "sentiment": round(
                        float(sentiment),
                        3
                    )

                })


                # Only keep 5 headlines
                if len(latest_news) >= 5:
                    break


    except Exception as e:

        print(
            "News API error:",
            e
        )


    # ==========================================
    # CALCULATE OVERALL NEWS SENTIMENT
    # ==========================================

    if sentiments:

        news_sentiment = float(
            np.mean(sentiments)
        )

    else:

        news_sentiment = 0.0


    # ==========================================
    # 12. 5-DAY FORECAST
    # ==========================================

    forecast_data = data.copy()

    forecast_prices = []


    for day in range(1, 6):

        latest_forecast = (
            forecast_data.iloc[-1]
        )


        forecast_current_price = float(
            latest_forecast["Close"]
        )


        future_features = pd.DataFrame(
            [[

                forecast_current_price,

                latest_forecast["Return"],

                latest_forecast["MA10"],

                latest_forecast["MA50"],

                latest_forecast["Volume"],

                latest_forecast["Volatility"],

                latest_forecast["Momentum"]

            ]],
            columns=features
        )


        # Safety check
        if future_features.isnull().values.any():

            predicted = forecast_current_price

        else:

            predicted = float(
                price_model.predict(
                    future_features
                )[0]
            )


        forecast_prices.append(
            round(
                predicted,
                2
            )
        )


        # --------------------------------------
        # CREATE NEW FORECAST ROW
        # --------------------------------------

        new_row = (
            latest_forecast.copy()
        )


        new_row["Close"] = predicted


        new_row["Return"] = (
            predicted
            / forecast_current_price
        ) - 1


        # --------------------------------------
        # MA10
        # --------------------------------------

        recent_prices = (
            forecast_data["Close"]
            .tail(9)
            .tolist()
        )


        recent_prices.append(
            predicted
        )


        new_row["MA10"] = np.mean(
            recent_prices
        )


        # --------------------------------------
        # MA50
        # --------------------------------------

        recent_prices_50 = (
            forecast_data["Close"]
            .tail(49)
            .tolist()
        )


        recent_prices_50.append(
            predicted
        )


        new_row["MA50"] = np.mean(
            recent_prices_50
        )


        # --------------------------------------
        # MOMENTUM
        # --------------------------------------

        if len(forecast_data) >= 10:

            old_price = (
                forecast_data["Close"]
                .iloc[-10]
            )


            if old_price != 0:

                new_row["Momentum"] = (
                    predicted
                    / old_price
                ) - 1


        # --------------------------------------
        # ADD FORECAST ROW
        # --------------------------------------

        forecast_data.loc[
            len(forecast_data)
        ] = new_row


    # ==========================================
    # 13. 30-DAY FORECAST
    # ==========================================

    forecast_30_data = (
        forecast_data.copy()
    )

    forecast_30_prices = []


    # We already generated days 1-5.
    # Generate days 6-30.

    for day in range(6, 31):

        latest_forecast = (
            forecast_30_data.iloc[-1]
        )


        forecast_current_price = float(
            latest_forecast["Close"]
        )


        future_features = pd.DataFrame(
            [[

                forecast_current_price,

                latest_forecast["Return"],

                latest_forecast["MA10"],

                latest_forecast["MA50"],

                latest_forecast["Volume"],

                latest_forecast["Volatility"],

                latest_forecast["Momentum"]

            ]],
            columns=features
        )


        # Safety check
        if future_features.isnull().values.any():

            predicted = (
                forecast_current_price
            )

        else:

            predicted = float(
                price_model.predict(
                    future_features
                )[0]
            )


        forecast_30_prices.append(
            round(
                predicted,
                2
            )
        )


        # --------------------------------------
        # NEW FORECAST ROW
        # --------------------------------------

        new_row = (
            latest_forecast.copy()
        )


        new_row["Close"] = predicted


        new_row["Return"] = (
            predicted
            / forecast_current_price
        ) - 1


        # --------------------------------------
        # MA10
        # --------------------------------------

        recent_prices = (
            forecast_30_data["Close"]
            .tail(9)
            .tolist()
        )


        recent_prices.append(
            predicted
        )


        new_row["MA10"] = np.mean(
            recent_prices
        )


        # --------------------------------------
        # MA50
        # --------------------------------------

        recent_prices_50 = (
            forecast_30_data["Close"]
            .tail(49)
            .tolist()
        )


        recent_prices_50.append(
            predicted
        )


        new_row["MA50"] = np.mean(
            recent_prices_50
        )


        # --------------------------------------
        # MOMENTUM
        # --------------------------------------

        if len(forecast_30_data) >= 10:

            old_price = (
                forecast_30_data["Close"]
                .iloc[-10]
            )


            if old_price != 0:

                new_row["Momentum"] = (
                    predicted
                    / old_price
                ) - 1


        # --------------------------------------
        # ADD ROW
        # --------------------------------------

        forecast_30_data.loc[
            len(forecast_30_data)
        ] = new_row


    # ==========================================
    # 14. FINAL 30-DAY PRICE
    # ==========================================

    if forecast_30_prices:

        predicted_30_day_price = float(
            forecast_30_prices[-1]
        )

    else:

        predicted_30_day_price = (
            original_current_price
        )


    expected_30_day_change = (
        predicted_30_day_price
        - original_current_price
    )


    expected_30_day_change_percent = (
        expected_30_day_change
        / original_current_price
    ) * 100


    # ==========================================
    # 15. OVERALL PREDICTION
    # ==========================================

    # Strong positive conditions
    if (
        expected_30_day_change_percent > 3
        and confidence >= 0.55
        and news_sentiment >= 0.05
    ):

        overall_prediction = "BULLISH"


    # Strong negative conditions
    elif (
        expected_30_day_change_percent < -3
        and confidence >= 0.55
        and news_sentiment <= -0.05
    ):

        overall_prediction = "BEARISH"


    # Everything else
    else:

        overall_prediction = "NEUTRAL"


    # ==========================================
    # 16. PRICE HISTORY
    # ==========================================

    history = []


    for index, row in data.tail(100).iterrows():

        try:

            price = float(
                row["Close"]
            )


            history.append({

                "date":
                    index.strftime(
                        "%Y-%m-%d"
                    ),

                "price":
                    round(
                        price,
                        2
                    )

            })

        except Exception:

            continue


    # ==========================================
    # 17. RETURN RESULTS
    # ==========================================

    return {

        # --------------------------------------
        # BASIC
        # --------------------------------------

        "ticker":
            ticker.upper(),


        # --------------------------------------
        # PRICE
        # --------------------------------------

        "current_price":
            round(
                original_current_price,
                2
            ),


        "predicted_price":
            round(
                predicted_price,
                2
            ),


        "expected_change":
            round(
                expected_change,
                2
            ),


        "expected_change_percent":
            round(
                expected_change_percent,
                2
            ),


        # --------------------------------------
        # DIRECTION
        # --------------------------------------

        "direction":
            direction,


        "confidence":
            round(
                float(
                    confidence * 100
                ),
                2
            ),


        "up_probability":
            round(
                float(
                    probabilities[1] * 100
                ),
                2
            ),


        "down_probability":
            round(
                float(
                    probabilities[0] * 100
                ),
                2
            ),


        # --------------------------------------
        # 5-DAY FORECAST
        # --------------------------------------

        "forecast":
            forecast_prices,


        # --------------------------------------
        # 30-DAY FORECAST
        # --------------------------------------

        "forecast_30_day":
            forecast_30_prices,


        "predicted_30_day_price":
            round(
                predicted_30_day_price,
                2
            ),


        "expected_30_day_change_percent":
            round(
                expected_30_day_change_percent,
                2
            ),


        # --------------------------------------
        # OVERALL PREDICTION
        # --------------------------------------

        "overall_prediction":
            overall_prediction,


        # --------------------------------------
        # HISTORY
        # --------------------------------------

        "history":
            history,


        # --------------------------------------
        # NEWS
        # --------------------------------------

        "news_sentiment":
            round(
                news_sentiment,
                3
            ),


        "news":
            latest_news

    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    result = predict_stock("AAPL")

    print(result)