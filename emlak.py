import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import pandas_profiling
import seaborn as sns
from sklearn.externals import joblib


    #Sidebars
    menuler = ["Hakkinda","Tahmin Yap!","Turgutlu Istatistlik"]
    secim = st.sidebar.selectbox("Menu Seciniz...",menuler)

    if secim == 'Tahmin Yap!':

        #Text/Title
        st.title("Turgutlu Emlak Tahmini")
        emlak = pd.read_csv('emlak.csv', encoding = 'iso-8859-9',sep=';') # veri setinin okunması

        X = emlak.iloc[:, 1:9].values
        y = emlak.iloc[:, 9].values
        y = y.reshape(-1, 1) 
        # Veri setinin bölümlendirilmesi


        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        # veri setinin test ve train olarak ayrılması 

        

        #Mahalle ve Isıtma Tipleri diziye atanmıştır
        mahalle_dizi = ['Cumhuriyet','Ataturk','Yildirim','Selvilitepe',
        'Subasi','Turgutlar','Ozyurt','Ergenekon',
        'Yeni','Yigitler','Acarlar','Albayrak',
        'Kurtulus','Mustafa Kemal','Sehitler','Yedi Eylul'
        'Istiklal','Dalbahce','Yilmazlar']
        isitma_dizi = ["Kombi Dogalgaz","Klima","Soba","Merkezi"]
        
        #Aranan mahallenin indexi dondurulmustur
        def linear_search(alist, key):
            for i in range(len(alist)):
                if alist[i] == key:
                    return i
            return -1
        
        #SelectBox
        mahalle = st.selectbox("Mahalle",mahalle_dizi)
        mahalle_index = linear_search(mahalle_dizi,mahalle) + 1
        #st.write(" Mahalle :",mahalle_index)
        
        #metrekare
        metrekare = st.slider("Metrekare(net)",50,200)

        #oda
        oda = st.slider("Oda Sayisi(+1)",1,5)

        #SelectBox
        isitma = st.selectbox("Isitma Tipi",["Kombi Dogalgaz","Klima","Soba","Merkezi"])
        isitma_index = linear_search(isitma_dizi,isitma) + 1
        #st.write(" İsitma :",isitma_index)

        #Bina Yaşı
        bina_Yasi = st.slider("Bina Yasi",0,30)

        #Bulunduğu Kat
        bulundugu_Kat = st.slider("Bulundugu Kat",0,15)

        #Toplam Kat
        toplam_Kat = st.slider("Bina Toplam Kat",0,15)

        #Banyo Sayısı
        banyo_sayi = st.slider("Banyo Sayisi",1,2)

        #Eşya Durumu
        esya_durum = st.slider("Esya Durum(Esyali:1 Esyasiz:0)",0,1)
        
        columns_model = joblib.load('model_columns.pkl')#modeldeki sutunların çekilmesi
        
        res = pd.DataFrame(columns=columns_model,data = 
        {'Banyo_Sayisi':[banyo_sayi],'Bina_KatSayisi':[toplam_Kat],'Bina_Yasi':[bina_Yasi],
         'Bulundugu_Kat':[bulundugu_Kat],'Esya_Durumu':[esya_durum],'Isitma_Tipi':[isitma_index],
          'Mahalle':[mahalle_index],'Metrekare(net)':[metrekare],'Oda_Sayisi(+1)':[oda]})
        
        #SelectBox
        model = st.selectbox("Model",["Desicion Tree","Random Forest","SVR"])
        st.write(" Model :",model)
        
        if model == 'Desicion Tree':
            dt = joblib.load('desicion_tree_model.pkl')
            prediction = dt.predict(res)
        
        if model == 'Random Forest':
            rf = joblib.load('random_forest_model.pkl')
            prediction = rf.predict(res)
            
        if model == 'SVR':
            svr = joblib.load('svr_model.pkl')
            prediction = svr.predict(res)

        #Tahmin değerini stringe dönüştürme
        prediction = np.round(prediction)#yuvarlama
        tahmin_sonuc = str(prediction).strip('[]')#parantezerden kurtulma
        
        #Buttons
        if st.button("Tahmin Et"):
            st.write("Emlak Degeri :",tahmin_sonuc,"000 TL")
            #Ballons
            st.balloons()

    if secim == 'Turgutlu Istatistlik':
        #Görselleştirme
        
        
    if secim == 'Hakkinda':
        st.write("Hamdi Damar 172802029")
        st.write("Cihan Cifci 172802019")
        st.write("Tarafindan Gelistirilmistir.!")





