# -*- coding: utf-8 -*-
"""Submission_DiwaPrasetyo.ipynb

Diwa Prasetyo

# Data Preparation
"""

import pandas as pd
import tensorflow as tf
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

df = pd.read_csv('/content/drive/MyDrive/BBC-News_Train.csv')
df

df = df.drop(columns = 'ArticleId')
df

category = pd.get_dummies(df.Category)
df_baru = pd.concat([df, category], axis=1)
df_baru = df_baru.drop(columns='Category')
df_baru = df_baru.replace({True:1, False:0})
df_baru

df_baru["Text"] = df_baru["Text"].str.lower()
def remove_html_tags(text):

    html_tag_pattern = re.compile("<.*?>")
    cleaned_text = html_tag_pattern.sub(r"", text)
    return cleaned_text

df_baru["Text"] = df_baru["Text"].apply(remove_html_tags)
df_baru

def remove_urls(text):

    url_pattern = re.compile(r"https?://\S+|www\S+")
    cleaned_text = url_pattern.sub(r"", text)
    return cleaned_text

df_baru["Text"] = df_baru["Text"].apply(remove_urls)

def remove_punctuations(text):
    punctuation_chars = string.punctuation
    cleaned_text = text.translate(str.maketrans("", "", punctuation_chars))
    return cleaned_text

df_baru["Text"] = df_baru["Text"].apply(remove_punctuations)
df_baru

"""Data lebih dari 1000 sampel"""

text = df_baru['Text'].values
label = df_baru[['business', 'entertainment', 'politics', 'sport', 'tech']].values

"""# Modeling"""

#validation test 0.2 dari total
text_latih, text_test, label_latih, label_test = train_test_split(text,label, test_size=0.2)

#Penggunaan Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text_latih)
tokenizer.fit_on_texts(text_test)

sekuens_latih = tokenizer.texts_to_sequences(text_latih)
sekuens_test = tokenizer.texts_to_sequences(text_test)

padded_latih = pad_sequences(sekuens_latih)
padded_test = pad_sequences(sekuens_test)

#Penggunaan model sequntial, Embedding dan LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#menggunakan callbacks
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') and logs.get('val_accuracy')>0.85):
      print("\nAkurasi telah mencapai >90%!")
      self.model.stop_training = True
callbacks = myCallback()

#akurasi train dan validation diatas 85% ( bisa mencapai 90% jika dilatih lagi )
num_epochs = 50
history = model.fit(padded_latih, label_latih, epochs=num_epochs, batch_size = 25,
                    validation_data=(padded_test, label_test), verbose=2, callbacks=[callbacks])

#memvisualisasikan plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
