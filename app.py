# %%
import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import urllib.request
from utils import *
import csv
import pandas as pd
import requests
from ultralytics import YOLO
from rembg import remove
import cv2
import numpy
import OpenGL


# %%
st.set_page_config(
    page_title="Glue Extractor",
    page_icon="🔧",
)

# %%
html_temp = '''
<div style = padding bottom: 20px; padding-left: 5px; padding right: 5px">
<center><h1>Glue Path</h1></center>
</div>
'''
st.markdown(html_temp, unsafe_allow_html=True)

# %%
html_temp = '''
<div>
<h2></h2>
<center><h3>Please upload Image</h3></center>
</div>
'''
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select','Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file)
        st.image(image, width = 640, caption = 'Uploaded Image')
        

def preprocess(img):
  output= remove(img)
  return output

  

try:
    if image is not None:
       if st.button('Extract'):
          img = preprocess(image)
          model = YOLO(r"best.pt")
          results = model(img)
          for result in results:
              b=result.names
              for j,mask in enumerate(result.masks.data):
                  mask = (mask.numpy() * 255).astype(np.uint8)  # Convert to uint8
                  mask_image = Image.fromarray(mask)
                  st.image(mask, width = 640, caption = b)
                  cv2.imwrite("wout.png",mask)


except Exception as e:
  st.info("Hello User!!")
  pass
