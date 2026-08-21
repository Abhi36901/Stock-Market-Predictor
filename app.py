from flask import Flask, render_template, request

from predictor import predict_stock


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():

    result = None

    ticker = request.args.get("ticker")

    print("TICKER RECEIVED:", ticker)

    if ticker:

        ticker = ticker.strip().upper()

        result = predict_stock(ticker)

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)