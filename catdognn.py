# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 01:22:21 2018

@author: elliott
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


"""
#to train nn:
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
print("done")
training_set = train_datagen.flow_from_directory('C:\\Users\\elliott\\Desktop\\Image Classification (cat & dog)\\training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('C:\\Users\\elliott\\Desktop\\Image Classification (cat & dog)\\test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

classifier.fit_generator(training_set, steps_per_epoch = 250, epochs = 30, validation_data = test_set, validation_steps = 63)


classifier.save('my_model30.h5')"""

from keras.models import load_model
classifier = load_model('my_model.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:\\Users\\elliott\\Desktop\\Image Classification (cat & dog)\\test_set\\cats\\cat.4187.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
#training_set.class_indices
prediction = ''
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

#try rebooting computer
