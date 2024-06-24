import numpy as np
import seaborn as sns
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

from tensorflow.keras import layers
from tensorflow.keras import models

from PIL import Image as PImage

import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#check it
def filter_class(tuple, reduceClass):
    if (tuple[1] not in set(reduceClass)):
        return tuple
    else: return

def repl_labelMap(sourceLabel, destLabels:list):
    if (sourceLabel == 1):
        return destLabels[0]
    elif (sourceLabel == 2):
        return destLabels[1]
    else: return -1
    #if (sourceLabel == 0):
    #    return destLabels[2]

def chosenTransposeMap(x, label, labelToTranspose):
    if (label == labelToTranspose):
        return tf.transpose(x)
    else: return x

def displayDataset(ds:tf.data.Dataset, dsname=''):
    # Dataset inside
    k = 0
    for elem in ds:  # .as_numpy_iterator():
        print(dsname + ' ds el', k, ': ', elem)
        k = k + 1

img = PImage.open('GRU.png')
plt.xlim(0, 0)  # x-axis limits
plt.ylim(0, 0)
plt.imshow(img)
#WARNING IN DATASET Relaxation marked as 0, Phys. grip as 1, ....., Ment. ungrip as 4
#It was to make learning possible. Know that while using repl_labelMap
abs_commands = {1: 'Relaxation', \
                2: 'Физическое сжатие', \
                3: 'Физическое разжатие', \
                #4: b'Mental_grip', \
                #5: b'Mental_ungrip'
               }

commands = {2: 'Физическое сжатие', \
            3: 'Физическое разжатие'}

ds = tf.data.Dataset.load('ConcatDS6-') #'Full23DsLSTM'
#train_ds = tf.data.Dataset.load('TrainHugeDSLSTM')
#valid_ds = tf.data.Dataset.load('ValidHugeDSLSTM')
#test_ds = tf.data.Dataset.load('TestHugeDSLSTM')
sample_ds = tf.data.Dataset.load('sample23DsLSTM')


#ds = ds.map(lambda x,y: (tf.transpose(x), y))
#ds = ds.map(lambda x,y: (chosenTransposeMap(x, y, 2), y))

displayDataset(ds, 'fullds el')
ds = ds.map(lambda x,y: (x, repl_labelMap(y, [0,1])))





#train valid test division
leftvalue = int(ds.cardinality().numpy() * 0.8)
rightvalue = ds.cardinality().numpy() * (1-0.8)

valCount = int(rightvalue // 2)
if (rightvalue % 2) != 0:
    valCount = valCount + 1

testCount = int(rightvalue // 2)

train_ds, valtes_ds = tf.keras.utils.split_dataset(ds, leftvalue, int(rightvalue)+1,
                                                   shuffle=True)
valid_ds, test_ds = tf.keras.utils.split_dataset(valtes_ds, valCount, testCount)
#End train valid test

buffer_size = train_ds.cardinality() #21
train_ds.shuffle(buffer_size)
#Scaling the Data
'''
To help the LSTM model to converge faster it is important
 to scale the data. It is possible that large values
 in the inputs slows down the learning. 
 We are going to use StandardScaler from sklearn library
 to scale the data.
'''

#build Neural Network
#Training parameters

lrng_rate = 1e-4 #for adam
#display_step = 1


# Network parameters

n_hidden = n_neurons = 64 #32
n_layers = 2  # 1 layer cannot figure out nonlinearity

num_labels = len(commands)
n_classes = num_labels
#n_inputs = 52 #n_colomns from idoxy
#n_steps = 40

def datasetToNpXY(specLabelsDataset):
    X_specs = specLabelsDataset.map(map_func=lambda spec, label: spec)
    Y_labels = specLabelsDataset.map(map_func=lambda spec, label: label)
    X_specs = np.array(list(X_specs.as_numpy_iterator()))
    Y_labels = np.array(list(Y_labels.as_numpy_iterator()))
    #print('X-factor ', type(X_specs))
    #Y_labels.label_encoder.transform(Y_labels)
    #Y_labels = np_utils.to_categorical(Y_labels)
    #print('X-factor ', X_specs)

    return X_specs, Y_labels

trainX, trainY = datasetToNpXY(train_ds)
validX, validY = datasetToNpXY(valid_ds)
testX, testY = datasetToNpXY(test_ds)


#Dataset inside
'''
k = 0
for elem in train_ds:  # .as_numpy_iterator():
    print('train_ds el', k, ': ', elem)
    k = k + 1
'''
#define input_shape

batch_size = 64
#trainSpec_ds = trainSpec_ds.batch(batch_size)
preBatchTrain_ds = train_ds



#also Dataset inside
for step, (ts_command, label) in enumerate(preBatchTrain_ds.take(train_ds.cardinality())):
    preBatchShape = ts_command.shape
print('preBatch shape:', preBatchShape)


train_ds = train_ds.batch(batch_size)
valid_ds = valid_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
#Добавьте Dataset.cache и Dataset.prefetch , чтобы уменьшить задержку чтения при обучении модели:
#train_ds = train_ds.cache().prefetch(AUTOTUNE)
#valid_ds = valid_ds.cache().prefetch(AUTOTUNE)
#test_ds = test_ds.cache().prefetch(AUTOTUNE)

for ts_commandb, _ in train_ds.take(1):
    batchedShape = ts_commandb.shape
print('Batched shape:', batchedShape)


#inputs = np.random.random((32, 10, 8))
#lstm = layers.LSTM(n_hidden)
#output = lstm(ts_commandb)
#print('outsh ', output.shape)

lstm = layers.LSTM(n_hidden)
output = lstm(ts_commandb)
print('outsh ', output[0].shape)

model = models.Sequential()
model.add(layers.Input(shape=preBatchShape))
norm_layer = layers.Normalization()
norm_layer.adapt(data=preBatchTrain_ds.map(map_func=lambda spec, label: spec)) #remove labels
model.add(norm_layer)
model.add(layers.LSTM(units=n_hidden, return_sequences=True, dropout=0.3))
model.add(layers.LSTM(units=n_hidden, return_sequences=False, dropout=0.3))
#model.add(layers.Bidirectional(layer=layers.LSTM(units=n_neurons, return_sequences=True, dropout=0.3)))
#model.add(layers.Bidirectional(layer=layers.LSTM(units=n_neurons, return_sequences=False, dropout=0.3)))
#model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(num_labels, activation='relu'))


model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lrng_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']#, 'Precision', 'Recall'],
)

EPOCHS = 10
history = model.fit(
    train_ds,#trainX, trainY,
    validation_data=valid_ds,#(validX, validY),
    epochs=EPOCHS,
    batch_size=batch_size,
    #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

#model.save()

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
  plt.title(f'Predictions for "{comm_labels[label[0]]}"')
  plt.show()



##############################  MAIN CODE END  ####################################
