# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# load data 
df_train = pd.read_table(train_file_path, header=0, names=['indicates', 'text'], usecols=['indicates', 'text'])
df_test = pd.read_table(test_file_path, header=0, names=['indicates', 'text'], usecols=['indicates', 'text'])

# convert indicates column to numeric 
df_train['indicates'] = df_train['indicates'].replace("ham", 0)
df_test['indicates'] = df_test['indicates'].replace("ham", 0)
df_train['indicates'] = df_train['indicates'].replace("spam", 1)
df_test['indicates'] = df_test['indicates'].replace("spam", 1)

# convert arrays to objects
train_data = tf.data.Dataset.from_tensor_slices((df_train['text'].values, df_train['indicates'].values))
test_data = tf.data.Dataset.from_tensor_slices((df_test['text'].values, df_test['indicates'].values))

# shuffle data and create batches 
BUFFER_SIZE = 10000
BATCH_SIZE = 32

train_dataset = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# get vocab size 
tokenizer =tfds.deprecated.text.Tokenizer()

vocab = set()

for text_tensor, _ in train_data.concatenate(test_data):
  token = tokenizer.tokenize(text_tensor.numpy())
  vocab.update(token)

vocab_size = len(vocab)
print(vocab_size)

# create text encoder
encoder = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size
)

# adabt train dataset to encoder
encoder.adapt(train_dataset.map(lambda text, label: text))

# creating the model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# train the model
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)

# evaluate the model
loss, acc = model.evaluate(test_dataset)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
  predictions = model.predict(np.array([pred_text])).tolist()
  result = []
  if predictions[0][0] < 0:
    result.append(0)
    result.append("ham")
  else:
    result.append(1)
    result.append("spam")
  return result

pred_text = "how are you doing today"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
