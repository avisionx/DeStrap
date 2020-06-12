import os
import shutil
import hashlib
import numpy as np
import cv2

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from PIL import Image

VOCAB         = ", { } small-title text quadruple row btn-inactive btn-orange btn-green btn-red double <START> header btn-active <END> single".splitlines()[0]
MAX_LENGTH    = 150

class Dataset:

    def __init__(self):
        self.tokenizer = Tokenizer(filters='', split=" ", lower=False)
        self.tokenizer.fit_on_texts([VOCAB])
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def create_generator(self, data_input_path):
        
        total_sequences = 0
        img_features, text_features = self.load_data(data_input_path)
        
        for text_set in text_features: 
            total_sequences += len(text_set.split())

        steps_per_epoch = total_sequences // 64
        data_gen = self.data_generator(text_features, img_features)

        return data_gen, steps_per_epoch

    def process_data_for_generator(self, texts, features):
        sequences = self.tokenizer.texts_to_sequences(texts)
        X, y, image_data = [], [], []
        for img_no, seq in enumerate(sequences):
            seq_len = len(seq)
            for i in list(range(1, seq_len)):
                in_seq = pad_sequences([seq[: i]], maxlen=MAX_LENGTH)
                out_seq = to_categorical([seq[i]], num_classes=self.vocab_size)
                X.append(in_seq[0][-48:])
                y.append(out_seq[0])
                image_data.append(features[img_no])
        return np.array(image_data), np.array(X), np.array(y)

    def load_data(self, data_input_path):

        text, images = [], []
        all_filenames = os.listdir(data_input_path)
        
        for filename in all_filenames:
            
            file_extension = filename[-3:]
            file_to_open = data_input_path + '/' + filename

            if(file_extension == 'gui'):
                file = open(file_to_open, 'r')
                texts = file.read()
                file.close()
                text.append(texts)

            elif(file_extension == "npz"):
                image = np.load(file_to_open)
                images.append(image['features'])
        
        for i in range(len(text)):
            syntax = '<START>' + text[i] + '<END>'
            syntax = ' '.join(syntax.split()).replace(',', ' ,')
            text[i] = syntax

        images = np.array(images, dtype=float)
        return images, text

    def data_generator(self, text_features, img_features):
        while(True):
            for i in range(0, len(text_features)):
                Ximages, XSeq, y = [], [], []
                for j in range(i, min(i + 1, len(text_features))):
                    in_img, in_seq, out_word = self.process_data_for_generator([text_features[j]], [img_features[j]])
                    for l in range(len(in_seq)):
                        XSeq.append(in_seq[l])
                    for m in range(len(in_img)):
                        Ximages.append(in_img[m])
                    for n in range(len(out_word)):
                        y.append(out_word[n])
                yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]

class ImagePreprocessor:

    def build_image_dataset(self, data_input_folder, augment_data=True):

        print("Converting images from {} into arrays, augmentation: {}".format(data_input_folder, augment_data))
        resized_img_arrays, sample_ids = self.get_resized_images(data_input_folder)

        if augment_data:
            self.augment_and_save_images(resized_img_arrays, sample_ids, data_input_folder)
        else:
            self.save_resized_img_arrays(resized_img_arrays, sample_ids, data_input_folder)

    def save_resized_img_arrays(self, resized_img_arrays, sample_ids, output_folder):
        count = 0
        zipped = zip(resized_img_arrays, sample_ids)
        attri = "features"
        for img_arr, sample_id in zipped:
            npz_filename = "{}/{}.npz".format(output_folder, sample_id)
            np.savez_compressed(npz_filename, features=img_arr)
            retrieve = np.load(npz_filename)[attri]
            assert np.array_equal(img_arr, retrieve)
            count += 1
        if count>=1:    
            print("Saved down {} resized images to folder {}".format(count, output_folder))
            del resized_img_arrays
        return None    

    def augment_and_save_images(self, resized_img_arrays, sample_ids, data_input_folder):
        datagen = ImageDataGenerator( width_shift_range=0.05, height_shift_range=0.05,rotation_range=2, zoom_range=0.05)
        count = 0
        # temp.data = datagen.copy()
        keras_generator = datagen.flow(resized_img_arrays, sample_ids, batch_size=1)
        attri = "features"
        for i in range(len(resized_img_arrays)):
            img_arr, sample_id = next(keras_generator)
            if(img_arr.all!=None):
                img_arr = np.squeeze(img_arr)
            npz_filename = "{}/{}.npz".format(data_input_folder, sample_id[0])
            # temp_img = img_arr.astype('uint8')
            # im = Image.fromarray(temp_img)
            np.savez_compressed(npz_filename, features=img_arr)
            retrieve = np.load(npz_filename)[attri]
            assert np.array_equal(img_arr, retrieve)
            count += 1
        if(count>=1):    
            print("Saved down {} augmented images to folder {}".format(count, data_input_folder))
            del resized_img_arrays
        return None    

    def get_resized_images(self, pngs_input_folder):
        images = []
        labels = []
        all_files = os.listdir(pngs_input_folder)
        img_shape = (256,256,3,)
        png_files = [f for f in all_files if f.find(".png") != -1]
        
        for png_file_path in png_files:
            
            sample_id = png_file_path[:png_file_path.find('.png')]


            img_rgb = cv2.imread("{}/{}".format(pngs_input_folder, png_file_path))
            
            labels.append(sample_id)
            img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            
            bg_img = np.ones(shape=img_shape)

            img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
            bg_img *= 255
            img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
            
            new_shape = (200, 200)
            resized = cv2.resize(img_stacked, new_shape, interpolation=cv2.INTER_AREA)
            
            bg_img[26:226, 26:226,:] = resized
            resized_img_arr=  bg_img / 255

            images.append(resized_img_arr)


        return np.array(images), np.array(labels)
        