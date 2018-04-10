import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D, MaxPooling2D 
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import wandb
from wandb.wandb_keras import WandbKerasCallback

# logging code
run = wandb.init()
config = run.config
config.epochs = 100

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train/255
X_test = X_test/255

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Reshape((img_width, img_height, 1), input_shape=(img_width,img_height)))
#model.add(Dropout(0.4))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,3), activation='relu'))
#model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

print(model.summary())
# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbKerasCallback(validation_data=X_test, labels=labels), EarlyStopping(monitor='val_acc', min_delta=0.001,patience=3, verbose=1)])



