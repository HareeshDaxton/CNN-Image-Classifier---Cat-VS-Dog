from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64, 64, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_data = ImageDataGenerator(rescale = 1./255, shear_range= 0.3, zoom_range=0.3, horizontal_flip=True)
train_set = train_data.flow_from_directory('dataset/train', target_size=(64, 64), batch_size=4, class_mode='binary')

test_set = train_data.flow_from_directory('dataset/test', target_size=(64, 64), batch_size=4, class_mode='binary')

model.fit(train_set, steps_per_epoch=20, epochs=10, validation_data = test_set, validation_steps=3)

test_image = image.load_img('dataset/pred/img.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(train_set.class_indices)

if result[0][0] >= 0.5:
    pred = 'dog'
else:
    pred = 'cat'

print(pred)















































# model.add(Dense(units=100, activation='relu', input_shape=(5,)))
# model.add(Dense(88, activation='relu'))
# model.add(Dense(57, activation='sigmoid'))
#
# model.summary()