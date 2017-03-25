from flask import Flask, render_template, request
import summary

app = Flask("dating")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/results', methods=["GET"])
def get_results():
    return render_template("results.html", results=summary.summary(request.args["name"]))


@app.route("/explanation")
def explanation():
    return render_template("explanation.html")


@app.route("/wibbly_wobbly")
def wibbly():
    return render_template("wibbly_wobbly.html")

app.run(host="0.0.0.0", port=80)