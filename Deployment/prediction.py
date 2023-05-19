import streamlit as st
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

import pickle
# Load All Files

with open('t.pickle', 'rb') as file_1:
  t = pickle.load(file_1)
# Adding spanish to stopwords
spanish = stopwords.words('spanish')
additional_stopwords = []
for i in spanish:
  additional_stopwords.append(i)

# Adding french to stopwords
french = stopwords.words('french')
for i in french:
  additional_stopwords.append(i)

# Adding german to stopwords
german = stopwords.words('german')
for i in german:
  additional_stopwords.append(i)

# Setting stopwords with english as default language
stopwords = list(set(stopwords.words('english')))
for i in additional_stopwords:
  stopwords.append(i)

lemmatizer = WordNetLemmatizer()
def text_processing(text):

  # Converting all text to Lowercase
  text = text.lower()

  # Removing Unicode Characters
  text = re.sub("&#[A-Za-z0-9_]+", " ", text)

  # Removing punctuation
  text = text.translate(str.maketrans('', '', string.punctuation))
  
  # Removing Whitespace
  text = text.strip()

  # Removing emoji
  text = re.sub("[^A-Za-z\s']", " ", text)

  # Removing double space
  text = re.sub("\s\s+" , " ", text)
        
  # Tokenizing words
  tokens = word_tokenize(text)

  # Removing Stopwords
  text = ' '.join([word for word in tokens if word not in stopwords])

  # Lemmatizer
  text = lemmatizer.lemmatize(text)

  return text


model_gruimp = load_model('best_model.h5', compile=False)

def run():
  with st.form(key='Amazon_Customer_Review'):
      st.title('Amazon Customer Review')
      image = Image.open('amazon1.jpg')
      st.image(image)
      review_title = st.text_input('Title',value='Mystery at Walt Disney World')
      review_text = st.text_input('Comments',value='Book arrived in good condition, but took quite a bit longer to arrive than normal. Overall, I am very satisfied.')

      submitted = st.form_submit_button('Submit')

  df_inf = {
      'review_title': review_title,
      'review_text': review_text,

  }

  df_inf = pd.DataFrame([df_inf])
  # Data Inference
  df_inf_copy = df_inf.copy()
  # Applying all preprocessing in one document

  df_inf_copy['review_processed'] = df_inf_copy['review_text'].apply(lambda x: text_processing(x))
  st.dataframe(df_inf_copy)
  # Transform Inference-Set 
  df_inf_transform = df_inf_copy.review_processed
  df_inf_transform = t.texts_to_sequences(df_inf_transform)
  # Padding the dataset to a maximum review length in words

  df_inf_transform = pad_sequences(df_inf_transform, maxlen=102)


  if submitted:
      # Predict using Neural Network
      y_pred_inf = model_gruimp.predict(df_inf_transform)
      y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
      #st.write('# Is the customer at risk of churning ? :thinking_face:')
      if y_pred_inf == 0:
         st.subheader('Negative Feedback')
         st.write('Dear valued customer,')
         st.write('Thank you for taking the time to share your experience with us. We are sorry to hear that you were not satisfied with your recent purchase from Amazon. We take all feedback seriously and strive to provide the best possible shopping experience for our customers.')
         st.write('We apologize for any inconvenience caused and would like to make it right. Please let us know what specifically went wrong with your order and we will do everything we can to rectify the situation. We value your satisfaction and would appreciate the opportunity to earn back your trust.')
         st.write('Thank you again for bringing this to our attention. We hope to hear from you soon and look forward to the opportunity to serve you better in the future.')
         st.write('Best regards,')
         st.write('The Amazon Customer Service Team')
      else:
         st.subheader('Positive Feedback')
         st.write('Dear valued customer,')
         st.write('Thank you so much for taking the time to provide us with your positive feedback regarding your recent purchase from Amazon. We are thrilled to hear that you had a great shopping experience with us and that our product met your expectations.')
         st.write('At Amazon, we always strive to exceed our customers expectations and provide the best possible service. Your satisfaction is our top priority and we are delighted to know that we have achieved this with your recent purchase.')
         st.write('Your kind words have made our day and will serve as a motivation for us to continue delivering exceptional service to our customers. Thank you for choosing Amazon and we hope to have the pleasure of serving you again in the near future.')
         st.write('Best regards,')
         st.write('The Amazon Customer Service Team')

if __name__ == '__main__':
    run()