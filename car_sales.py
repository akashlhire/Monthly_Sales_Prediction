#!/usr/bin/env python

from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import multiprocessing


class StreamlitApp:
  def __init__(self):
    self.filename = 'model.sav'
    self.model = joblib.load(self.filename)

  def train_model(self):
    path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
    df = pd.read_csv(path,header=0)

    df.columns = ['ds','y']
    df['ds'] = pd.to_datetime(df['ds'])
    train = df.drop(df.index[-12:])

    model = Prophet()
    model.fit(train)
    filename = 'model.sav'
    joblib.dump(model,filename)


  def predict_sales(self):
    n = int(st.slider('',1,12,6))
    future = list()
    for i in range(1,n+1):
      date = '1969-%02d' % i
      future.append([date])
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds'] = pd.to_datetime(future['ds'])
        
    forecast = self.model.predict(future)
    y_pred = forecast['yhat'].values
    fig, ax = plt.subplots()
    ax.plot(future['ds'],y_pred, label='Predicted')
    ax.set_ylabel('Sales') 
    ax.set_xlabel('Months')
    ax.legend()
    #st.pyplot(fig)    
    st.plotly_chart(fig, use_container_width=True)

sa = StreamlitApp()
st.set_page_config( page_title = "Monthly Car Sales Forecasting Model",)
st.header("FB-Prophet Monthly Sales Forecasting Model")
st.subheader("Select the number of Months to be predicted")
#sa.train_model()
sa.predict_sales()





