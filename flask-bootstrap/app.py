from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd 
import numpy as np 



#paketler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

	#model okuma ve atama işlemleri
	dt = joblib.load('models/desicion_tree_model.pkl')
	columns_model = joblib.load('models/model_columns.pkl')

	cv = CountVectorizer()

	if request.method == 'POST':
		Mahalle = request.form['mahalle']
		Banyo = request.form['banyo']
		Katsayi = request.form['katsayi']
		Yas = request.form['yas']
		Kati = request.form['bulundugukat']
		Esya = request.form['esyadurum']
		Isitma = request.form['isitma']
		Oda = request.form['oda']
		Metrekare = request.form['metrekare']
		#data = [tahmin]
		#vect = cv.transform(data).toarray()
		#tahmin_deger = dt.predict(vect)

		res = pd.DataFrame(columns=columns_model,data = 
			{'Banyo_Sayisi':[Banyo],'Bina_KatSayisi':[Katsayi],'Bina_Yasi':[Yas],
			'Bulundugu_Kat':[Kati],'Esya_Durumu':[Esya],'Isitma_Tipi':[Isitma],
			'Mahalle':[Mahalle],'Metrekare(net)':[Metrekare],'Oda_Sayisi(+1)':[Oda]})
		prediction = dt.predict(res)
		prediction = np.round(prediction)#yuvarlama
        #tahmin_sonuc = str(prediction).strip('[]')#parantezerden kurtulma
        
		return render_template('results.html',prediction = prediction,name = res)

if __name__ == '__main__':
	app.run(debug=True)