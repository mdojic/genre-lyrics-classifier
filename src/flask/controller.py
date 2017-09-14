from flask import Flask, render_template, request, send_from_directory
from src.utils.language_detect import LanguageDetector
from src.clf.classify import Classify

app = Flask(__name__)

@app.route("/")
def index():
    return "e brt moj"

@app.route("/classify", methods=["GET"])
def classify():
    # return render_template("../../res/pages/classify.html")
    return send_from_directory(directory='../../res/pages', filename='classify.html')


@app.route("/get_genre", methods=["POST"])
def get_genre():
    print("Hit me baby one more time")
    req_lyrics = request.form.get("lyrics", "")
    print("Lyrics are: " + str(req_lyrics))

    lyrics_are_english = LanguageDetector.is_english(req_lyrics)
    if not lyrics_are_english:
        return "eng brt"

    genre = Classify.predict_lyrics_genre(req_lyrics)
    return genre

    # TODO:
    # 1. Check if lyrics are in english - return error if they aren't
    # 2. Check if lyrics are long enough - return error if they aren't
    # 3. Call classifier on lyrics - check for errors
    # 4. Return determined genres (or percentage of probability for each genre?)
    pass

@app.route('/js/<path:path>')
def send_js(path):
    print("Hit /js/path: " + path)
    return app.send_static_file("js/" + path)

if __name__ == "__main__":
    app.run()
