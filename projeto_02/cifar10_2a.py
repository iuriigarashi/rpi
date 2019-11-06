# In[]:
# Imports
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

# In[]:
# Definições da CNN
batch_size = 32
n_classes = 10
n_epochs = 15
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model_cifar10.h5'
cnn_last_layer_length = 256
keep_probability = 0.5
kernel_size = (3, 3)
pool_size = (2, 2)
# Não utilizado. Dataset importado direto pelo keras
# TRAIN_DIR = './kaggle/cifar10/train/'
# TEST_DIR = './kaggle/cifar10/test/'

# In[]:
# Data split
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape: ', x_train.shape)
print('train samples: ', x_train.shape[0])
print('test samples: ', x_test.shape[0])

# Converte para as classes categoricas
# Necessário para usar categorical_crossentropy
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# Input Shape
image_size = 32                    # Todas as imagens devem ter esse tamanho
nchannels = 3                      # Numero de canais na imagem
input_shape = (image_size, image_size, nchannels)

# In[]:
# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=kernel_size, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(keep_probability))

model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(keep_probability))

model.add(Flatten())
model.add(Dense(cnn_last_layer_length, activation='relu'))
model.add(Dropout(keep_probability/2))
model.add(Dense(n_classes, activation='softmax'))

# inicializa o RMSprop optimizer
rmsp = keras.optimizers.RMSprop(learning_rate=0.0001)

# Compile usando o otimizador RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=rmsp,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

# In[]:
# Grava model
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Model gravado em: %s ' % model_path)

# In[]:
# Score
scores = model.evaluate(x_test, y_test, verbose=1)
print('loss:', scores[0])
print('accuracy:', scores[1])

# %%
