#%%
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os, re, datetime, json, pickle
from tensorflow import keras
from keras.utils import pad_sequences, plot_model
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# %%
# 1. Data Loading

# read csv from url
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')

# %%
# 2. Data Inspection

# check for datattype
df.info()

# check for table summary
df.describe()

# Check for missing values
print(f'Missing values:\n{df.isna().sum()}')

# Check for complete duplicates
print(f'Duplicated values: {df.duplicated().sum()}')

# %%
# 3. Data Cleaning
# Drop complete duplicates
df.drop_duplicates(inplace=True)

# Separate feature to clean
text = df['text']

for idx,txt in text.items():
    text[idx] = re.sub('[^a-zA-Z]',' ',txt).lower()

#%%
# 4. Features Selection

# Extract label from dataframe
category = df['category']

#%% 
# 5. Data preprocesssing

# Feature tokenizer
vocab = 5000

token = Tokenizer(num_words=vocab, oov_token='<OOV>')
token.fit_on_texts(text)

train_sequences = token.texts_to_sequences(text)

# Padding + truncating
train_sequences = pad_sequences(train_sequences, maxlen=400, padding='post', truncating='post')

# Expand the labels and features into 2d array
train_sequences = np.expand_dims(train_sequences, -1)
train_labels = np.expand_dims(category, -1)

# OneHotEncoding
ohe = OneHotEncoder(sparse=False)
train_labels = ohe.fit_transform(train_labels)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_labels,shuffle=True,random_state=123)

# %%
# 6. Model Development
embed_dim = 128

model = Sequential()
model.add(Embedding(vocab, embed_dim))
model.add(LSTM(embed_dim,return_sequences=True))
model.add(LSTM(embed_dim))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)

#%%
# 7. Model compilation and training

#Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Define callbacks
LOG_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=LOG_PATH)

# Model training
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=tb)

# %%
# 8. Model evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Evaluate prediction
print('Classification Report:\n', classification_report(y_pred, y_true,zero_division=0))

# %% 
# 9. Model saving
# Save tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(token.to_json(), f)

# Save OHE
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# Save model
model.save('text-classification.h5')
