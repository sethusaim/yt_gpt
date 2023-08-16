from flask import Flask, jsonify, render_template, request

from src.embeddings import generate_embeddings
from src.qa import ask_query

app = Flask(__name__)
app.config["BOOTSTRAP_SERVE_LOCAL"] = True


@app.route("/admin", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        urls = request.form["urls"]
        embedding_index = request.form["embedding_index"]
        api_key = request.form["api_key"]
        index_name = request.form["index_name"]

        try:
            generate_embeddings(
                urls=urls,
                data_save_dir="our_dir",
                api_key=api_key,
                index_name=index_name,
            )

            return render_template(
                "generate_result.html",
                result_message=f"Generated embeddings for {urls} and saved with index name as {embedding_index}",
            )

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("admin.html")


@app.route("/", methods=["GET", "POST"])
def ask_yt_query():
    if request.method == "POST":
        query = request.form["query"]
        api_key = request.form["api_key"]
        index_name = request.form["index_name"]

        try:
            resp = ask_query(query=query, api_key=api_key, index_name=index_name)

            return render_template("result.html", response=resp)

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("query.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
