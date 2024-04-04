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
from ultralytics.utils.plotting import Annotator, colors

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
          im0 = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
          model = YOLO(r"best.pt")
          img = preprocess(im0)
          
          # Perform inference (prediction)
          results = model(img)
          names = model.model.names
          annotator = Annotator(im0, line_width=2)
          for results[0].masks is not None:
              clss = results[0].boxes.cls.cpu().tolist()
              masks = results[0].masks.xy
              for mask, cls in zip(masks, clss):
                  annotator.seg_bbox(mask=mask,mask_color=colors(int(cls), True),det_label=names[int(cls)])
              rgb_image = cv2.cvtColor(annotator.im, cv2.COLOR_BGR2RGB)
              st.image(rgb_image, caption="Instance Segmentation", use_column_width=True)
              #for j,mask in enumerate(result.masks.data):
              #    mask = (mask.numpy() * 255).astype(np.uint8)  # Convert to uint8
              #    mask_image = Image.fromarray(mask)
            
                  #img.write(img_cv)
                  #st.image(img, width = 640, caption= "Extracted Image")
                  #cv2.imwrite("wout.png",mask)


except Exception as e:
  st.info(f"Error accessing confidence score: {e}")
  pass
