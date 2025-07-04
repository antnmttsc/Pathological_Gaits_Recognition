import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import os

from Utils.Preprocessing import load_and_preprocess_data_sk, crop_data
import Utils.Constants as _c

# --- Custom Positional Encoding (needed when loading model)
@keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, max_len=100, **kwargs):
    super().__init__(**kwargs)
    self.max_len = max_len

  def build(self, input_shape):
    self.pos_embedding = self.add_weight(
      name="pos_embedding",
      shape=(self.max_len, input_shape[-1]),
      initializer="random_normal",
      trainable=True,
    )

  def call(self, x):
    length = tf.shape(x)[1]
    return x + self.pos_embedding[:length]

    
# --- Load Model
# @st.cache_resource
model = load_model("transformer_skeleton_model.h5", custom_objects={"PositionalEncoding": PositionalEncoding})

st.title("🕺 Pathological Gait Classification")
st.write("Upload a **.csv** file. The model will classify it into one of the 6 classes.")

# --- Upload CSV
uploaded_file = st.file_uploader("Upload a .csv file", type=["csv"])

if uploaded_file:
    
    temp_path = "temp_skeleton"
    with open(temp_path, "wb") as f:
      f.write(uploaded_file.read())

    try:

      # 1. Load and preprocess the data
      data = load_and_preprocess_data_sk(
        path=temp_path, 
        joints=[0,1,2,3,4,11,18,19,20,21,22,23,24,25], 
        clean_data=True, 
        norm=True,
        deploy=True
      )

      data = data.astype(np.float32)

      data = crop_data(
        data=data,
        target_size=50,
        crop_type='aggressive_center'
      )

      # Expand dimensions for batch
      data = np.expand_dims(data, axis=0)  # Shape: (1, 50, N)
      
      # 4. Predict
      prediction = model.predict(data)
      predicted_class = np.argmax(prediction, axis=1)[0]

      # Optional class labels
      labels = list(_c.convrt_gait_dict.keys())
      label_name = labels[predicted_class].upper()

      # Display result
      # st.success(f"🎯 Predicted Class: **{label_name}**")
      if label_name == "NORMAL":
        st.markdown(
          f"""
          <div style="background-color:#d4edda;padding:1em;border-radius:8px;color:#155724;border:1px solid #c3e6cb;">
              🎯 Predicted Class: <strong>{label_name}</strong>
          </div>
          """, unsafe_allow_html=True)
      else:
        st.markdown(
          f"""
          <div style="background-color:#f8d7da;padding:1em;border-radius:8px;color:#721c24;border:1px solid #f5c6cb;">
              🎯 Predicted Class: <strong>{label_name}</strong>
          </div>
          """, unsafe_allow_html=True)
      
      st.markdown("<br><br>", unsafe_allow_html=True)

      df = pd.DataFrame({'Class': labels,'Probability': prediction[0]})

      chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Class', sort=None, axis=alt.Axis(labelAngle=45)),
        y='Probability'
        ).properties(
          width=600,
          height=400,
          title=alt.TitleParams(
          text='Predicted Probabilities',
          anchor='middle',
          fontSize=18)
          )

      st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading the CSV: {e}")

#####################################
# python -m streamlit run my_app.py #
#####################################
