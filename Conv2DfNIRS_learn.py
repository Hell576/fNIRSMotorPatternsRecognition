import numpy as np
import seaborn as sns
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils

import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def add_dimension(tensor:tf.Tensor) -> tf.Tensor:
    tensor = tf.expand_dims(tensor, axis=-1)
    return tensor

def repl_labelMap(sourceLabel, destLabels:list):
    if (sourceLabel == 1):
        return destLabels[0]
    elif (sourceLabel == 2):
        return destLabels[1]
    else: return -1
    #if (sourceLabel == 0):
    #    return destLabels[2]

def displayDataset(ds:tf.data.Dataset, dsname=''):
    # Dataset inside
    k = 0
    for elem in ds:  # .as_numpy_iterator():
        print(dsname + ' ds el', k, ': ', elem)
        k = k + 1

#WARNING IN DATASET Relaxation marked as 0, Phys. grip as 1, ....., Ment. ungrip as 4
#It was to make learning possible. Know that while using repl_labelMap
abs_commands = {1: b'Relaxation', \
                2: b'Physical_grip', \
                3: b'Physical_ungrip', \
                #4: b'Mental_grip', \
                #5: b'Mental_ungrip'
               }

commands = {2: b'Physical_grip', \
            3: b'Physical_ungrip'}

ds = tf.data.Dataset.load('ConcatDS6-')
#train_ds = tf.data.Dataset.load('TrainHugeDSLSTM')
#valid_ds = tf.data.Dataset.load('ValidHugeDSLSTM')
#test_ds = tf.data.Dataset.load('TestHugeDSLSTM')
sample_ds = tf.data.Dataset.load('sample23DsLSTM')

displayDataset(ds, 'fullds el')


#ds = ds.map(lambda x,y: (tf.transpose(x), y))
ds = ds.map(lambda x,y: (x, repl_labelMap(y, [0,1])))
#prepare ds values as image dimension for Conv2D
ds = ds.map(lambda x,y: (add_dimension(x), y))

#train valid test division
leftvalue = int(ds.cardinality().numpy() * 0.8)
rightvalue = ds.cardinality().numpy() * (1-0.8)

valCount = int(rightvalue // 2)
if (rightvalue % 2) != 0:
    valCount = valCount + 1

testCount = int(rightvalue // 2)
#if (rightvalue % 2) != 0:
#    valCount = rightvalue // 2 + 1
#else: valCount = rightvalue // 2
train_ds, valtes_ds = tf.keras.utils.split_dataset(ds, leftvalue, int(rightvalue)+1,
                                                   shuffle=True)
valid_ds, test_ds = tf.keras.utils.split_dataset(valtes_ds, valCount, testCount)
#End train valid test

batch_size = 10
preBatchTrain_ds = train_ds
num_labels = len(commands)


#also Dataset inside
for step, (ts_command, label) in enumerate(preBatchTrain_ds.take(train_ds.cardinality())):
    preBatchShape = ts_command.shape
print('preBatch shape:', preBatchShape)


train_ds = train_ds.batch(batch_size)
valid_ds = valid_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

#define input_shape
for ts_commandb, _ in train_ds.take(1):
    batchedShape = ts_commandb.shape
print('Batched shape:', batchedShape) #input_shape ~= (5, None, 128)

#Добавьте Dataset.cache и Dataset.prefetch , чтобы уменьшить задержку чтения при обучении модели:
buffer_size = train_ds.cardinality()
train_ds = train_ds.cache().shuffle(buffer_size).prefetch(AUTOTUNE) #train.ds.cardinal
val_ds = train_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)


#build Neural Network
# Training parameters
lrng_rate = 1e-4 #for adam


# Network parameters

n_hidden = n_neurons = 32
n_layers = 2  # 1 layer cannot figure out nonlinearity
n_classes = num_labels
# n_inputs = 52 #n_colomns from idoxy
# n_steps = 40


# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=preBatchTrain_ds.map(map_func=lambda spec, label: spec))

#Conv2D
model = models.Sequential([
    layers.Input(shape=preBatchShape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # NO NEED to Normalize. Values are already between 0 and 1.
    norm_layer,
    #layers.LSTM(n_hidden, 3),
    #layers.LSTM(n_hidden, 3),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    #layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    #layers.GlobalAveragePooling1D(data_format='channels_last'),
    layers.Dense(num_labels),
])


#PseudoLSTM
'''
model = models.Sequential([
    layers.Input(shape=preBatchShape),
    # Downsample the input.
    #layers.Resizing(32, 32),
    # NO NEED to Normalize. Values are already between 0 and 1.
    norm_layer,
    layers.LSTM(n_hidden),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])
'''

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lrng_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


#METRIC SINGLE
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()


#TEST
test_tframe = []
test_labels = []

for tframe, label in test_ds:
  #print('audiodolby', tframe)
  test_tframe.append(tframe[0].numpy())
  test_labels.append(label[0].numpy())

test_tframe = np.array(test_tframe)
test_labels = np.array(test_labels)


y_pred = np.argmax(model.predict(test_tframe), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')


#some tips in ''' tips '''
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
'''

#METRICS DUAL
epochs_range = range(EPOCHS)

metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

plt.show()

#EVALUATION
model.evaluate(test_ds, return_dict=True)
y_pred = model.predict(test_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_ds.map(lambda s,lab: lab)), axis=0)


#CONFUSION MATRIX
comm_labels = list(commands.values())

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
#print(confusion_mtx)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=comm_labels,
            yticklabels=comm_labels,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

#REAL CHECK, WHAT NN UNDERSTANDS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#sample_ds = trioFilesToDataset(hbO_sampleFile, hbR_sampleFile,
#                          hdr_sampleFile, commands, equalTenslen=True)

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(comm_labels, tf.nn.softmax(prediction[0]))
  plt.title(f'Predictions for "{comm_labels[label[0]]}"') #data structure error
  plt.show()

##############################  MAIN CODE END  ####################################