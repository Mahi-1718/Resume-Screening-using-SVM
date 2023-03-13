import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import PyPDF2


app = Flask(__name__,template_folder = 'Template')
model = pickle.load(open("model.pkl", "rb"))


data = pd.read_csv('newresume.csv')
from sklearn.feature_extraction.text import CountVectorizer

requiredText= np.array(data["cleaned_resume"])
requiredTarget = np.array(data["Category"])

cv = CountVectorizer()
vv = cv.fit_transform(requiredText)



@app.route('/')
def home():
    return render_template('Index.html')


@app.route("/upload", methods=["POST"])
def predict():
    if 'pdfFile' in request.files:
        pdfFile = request.files['pdfFile']
        pdfReader = PyPDF2.PdfReader(pdfFile)
        numPages = len(pdfReader.pages)
        print('Number of pages:', numPages)
        for pageNum in range(numPages):
            page =  pdfReader.pages[pageNum]
            text = page.extract_text()

    data = cv.transform([text]).toarray()
    pre = str(model.predict(data))
    if pre == "Java Developer":
        prediction = "Selected"
    else:
        prediction = "Not Selected"

    return render_template('predict.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)