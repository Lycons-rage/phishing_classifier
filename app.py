from flask import Flask, render_template, jsonify, request

from src.pipeline.prediction_pipeline import CustomDataPreparation, PredictionPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/url_test", methods=["GET", "POST"])
def url_test():
    if request.method=="POST":
        try:
            url = str(request.form.get("url"))
            
        except Exception as e:
            return render_template("form.html", exception = e) 

        data_preparation = CustomDataPreparation(url=url)
        prepared_data = data_preparation.prepare_data()

        dataframe = data_preparation.get_data_as_dataframe(prepared_data)
        prediction = PredictionPipeline().predict_data(dataframe)

        return render_template("result.html", result = [prediction, dataframe])
        
    else:
        return render_template("form.html")
if __name__ == "__main__":
    app.run("0.0.0.0", 7000, debug=True)