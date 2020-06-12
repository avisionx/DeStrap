from keras.models import Model, Sequential, model_from_json

from keras.layers import Embedding, GRU, RepeatVector, LSTM, concatenate, Input, Dense
from keras.layers.core import Dense, Dropout, Flatten

from classes.Utils import *

from keras.layers.convolutional import Conv2D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop

VOCAB = ", { } small-title text quadruple row btn-inactive btn-orange btn-green btn-red double <START> header btn-active <END> single".splitlines()[0]

class DeStrap():

    def __init__(self):

        self.tokenizer = Tokenizer(filters='', split=" ", lower=False)
        self.tokenizer.fit_on_texts([VOCAB])
        self.vocab_size = len(self.tokenizer.word_index) + 1

        img_shape = (256,256,1,)
        image_model = Sequential()
        
        image_model.add(Conv2D(16, kernel_size =  (3, 3), padding='valid', activation='relu', input_shape=(256, 256, 1,)))
        image_model.add(Conv2D(16, kernel_size =(3,3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(32, kernel_size =(3,3), activation='relu', padding='same'))
        image_model.add(Conv2D(32, kernel_size =(3,3), activation='relu', padding='same'))
        image_model.add(Conv2D(32, kernel_size =(3,3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(64, kernel_size =(3,3), activation='relu', padding='same'))
        image_model.add(Conv2D(64, kernel_size =(3,3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(128, kernel_size =(3,3), activation='relu', padding='same'))
        image_model.add(Flatten())
        image_model.add(Dense(2048, activation='relu'))
        
        # image_model.add(Dropout(0.3))
        
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.35))
        image_model.add(RepeatVector(48))
        
        visual_input = Input(shape=img_shape)
        
        encoded_image = image_model(visual_input)
        language_input = Input(shape=(48,))
        language_model = Embedding(self.vocab_size, 50, input_length=48, mask_zero=True)(language_input)
        print("language_model1")
        print(language_model)
        language_model = GRU(128, return_sequences=True)(language_model)
        print("language_model2")
        print(language_model)
        language_model = GRU(128, return_sequences=True)(language_model)
        
        print("language_model3")
        print(language_model)

        model_list = [encoded_image,language_model]

        decoder = concatenate(model_list)
        self.lr = 0.0001
        print('decoder1')
        print(decoder)
        decoder = GRU(512, return_sequences=True)(decoder)
        print('decoder2')
        print(decoder)

        decoder = GRU(512, return_sequences=False)(decoder)
        print('decoder3')
        print(decoder)
        decoder = Dense(self.vocab_size, activation='softmax')(decoder)
        print('decoder4')
        print(decoder)

        self.model = Model(inputs=[visual_input, language_input], outputs=decoder)
        optimizer = RMSprop(self.lr, clipvalue=1.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def train(self, training_path, validation_path, epochs):

        training_generator, train_steps_per_epoch = Dataset().create_generator(training_path)
        validation_generator, val_steps_per_epoch = Dataset().create_generator(validation_path)

        self.model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs,  validation_steps=val_steps_per_epoch, steps_per_epoch=train_steps_per_epoch, verbose=1)
        
        model_json = self.model.to_json()
        
        with open("./save_model/model_json.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights("./save_model/weights.h5")