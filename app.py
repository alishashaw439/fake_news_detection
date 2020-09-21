import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
import os
import pickle
import flask
from newspaper import Article
import nltk
nltk.download('punkt')
import urllib
#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder="templates")

with open("model.pickle","rb") as handle:
    model = pickle.load(handle)


@app.route('/')
def main():
	return render_template("main.html")
#Receiving the input url from the user and using Web Scrappoing to extract the news content
@app.route("/predict", methods=["GET","POST"])
def predict():
	url=request.get_data(as_text=True)[5:]
	url=urllib.parse.unquote(url)
	article=Article(str(url))
	article.download()
	article.parse()
	article.nlp()
	news=article.summary
	#Passing the news article to the model and returning whether it is Fake or Real
	pred=model.predict([news])
	return render_template("main.html",prediction_text="The news is “{}”".format(pred[0]))
if __name__=="__main__":
	port=int(os.environ.get("PORT",5000))
	app.run(port=port,debug=True,use_reloader=False)
