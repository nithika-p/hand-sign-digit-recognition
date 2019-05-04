from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()

classifier.add(Convolution2D(32, (3,3), input_shape=(64,64,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=6,activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen =  ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train_data',
                                                target_size=(64,64),
                                                batch_size = 5,
                                                color_mode = 'grayscale',
                                                class_mode = 'categorical'
                                                )

test_set = test_datagen.flow_from_directory('data/test_data',
                                            target_size=(64,64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical'
                                            )

classifier.fit_generator(
    training_set,
    steps_per_epoch=3306,
    epochs=10,
    validation_data=test_set,
    validation_steps=30
)

model_json = classifier.to_json()
with open("model-bw.json","w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')
