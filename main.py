import os
import warnings
import librosa
# from librosa import display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.models import Model
# from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.regularizers import l
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout



warnings.filterwarnings('ignore')
dirWalk = './mainAud'
filepath = ".\\checkpoints\\weights-NEWdrops-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboardr = TensorBoard(log_dir="./checkpoints/logs/", write_images=True)
callbacks_list = [checkpoint, tensorboardr]

dirList = []
fileList = []
X = []


for root, dirs, files in os.walk(dirWalk):
    for dirr in dirs:
        dirList.append(dirr)


for root, dirs, files in os.walk(dirWalk):
    if len(files) != 0:
        fileList.append(files)
    else:
        continue


for i in range(0, len(dirList)):
    for j in fileList[i]:
        audd = "./mainAud/" + dirList[i] + "/" + j
        y, sr = librosa.load(audd, duration=2.97)
        y = np.pad(y, (0, 65489 - len(y)), 'constant', constant_values=0)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        X.append(ps)
X = np.array(X)
labels = np.zeros((len(X), 4))

countLab = 0
for i in range(0, len(fileList)):
    for j in fileList[i]:
        labels[countLab][i] = 1
        countLab += 1

# model splitting into train test and validation sets
# X = np.array([x.reshape((128, 128, 1)) for x in X])
X = X.reshape((len(X), 128, 128, 1))
xCut, xVal, yCut, yVal = tts(X, labels, test_size=0.1, random_state=1)
xTrain, xTest, yTrain, yTest = tts(xCut, yCut, test_size=0.2, random_state=1)

print(len(X), len(labels))
print(len(xTrain), len(yTrain))
print(len(xTest), len(yTest))
print(len(xVal), len(yVal))

# xTrain = np.array([x.reshape((128, 128, 1)) for x in xTrain])
# xTest = np.array([x.reshape((128, 128, 1)) for x in xTest])
# xVal = np.array([x.reshape((128, 128, 1)) for x in xVal])
# print(labels)

inputs = Input(shape=(128, 128, 1))

conv1 = Conv2D(24, (5, 5), input_shape=(128, 128, 1), activation='relu')(inputs)
maxPool1 = MaxPooling2D((4, 2))(conv1)

conv2 = Conv2D(48, (5, 5), activation='relu')(maxPool1)
maxPool2 = MaxPooling2D((4, 2))(conv2)

conv2 = Conv2D(24, (5, 5), activation='relu')(maxPool2)

conv3 = Conv2D(24, (2, 2), activation='relu', kernel_regularizer='l2')(conv2)

flat1 = Flatten()(conv3)

l1 = Dense(64, activation='relu', kernel_regularizer='l2')(flat1)
drop1 = Dropout(0.3)(flat1)

outputs = Dense(4, activation='softmax')(l1)
drop1 = Dropout(0.5)

model = Model(inputs=inputs, outputs=outputs)

# adam = Adam(lr=0.02)
model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
    metrics=['accuracy'])
history = model.fit(
    xTrain,
    yTrain,
    batch_size=128,
    epochs=30,
    callbacks=callbacks_list,
    validation_data=(xVal, yVal))

print(model.summary())

predictions = model.predict(xTest)
print(predictions)
print("\n")
print(yTest)

# bar chart
zHeight = predictions.sum(0)
zBar = ['clap', 'hiHat', 'kick', 'snare']
plt.bar(zBar, zHeight)
plt.show()

plt.plot(history.history['loss'])
plt.show()
