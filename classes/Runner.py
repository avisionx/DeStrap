from __future__ import absolute_import

import os
import numpy as np
import cv2

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from nltk.translate.bleu_score import sentence_bleu

from classes.Utils import ImagePreprocessor
from .Node import *

VOCAB = ", { } small-title text quadruple row btn-inactive btn-orange btn-green btn-red double <START> header btn-active <END> single".splitlines()[0]
DEFAULT_DSL_MAPPING = {
    "opening-tag": "{",
    "closing-tag": "}",
    "body": "<html>\n  <header>\n    <meta charset=\"utf-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n    <link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css\" integrity=\"sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u\" crossorigin=\"anonymous\">\n<link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css\" integrity=\"sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp\" crossorigin=\"anonymous\">\n<style>\n.header{margin:20px 0}nav ul.nav-pills li{background-color:#333;border-radius:4px;margin-right:10px}.col-lg-3{width:24%;margin-right:1.333333%}.col-lg-6{width:49%;margin-right:2%}.col-lg-12,.col-lg-3,.col-lg-6{margin-bottom:20px;border-radius:6px;background-color:#f5f5f5;padding:20px}.row .col-lg-3:last-child,.row .col-lg-6:last-child{margin-right:0}footer{padding:20px 0;text-align:center;border-top:1px solid #bbb}\n</style>\n    <title>Scaffold</title>\n  </header>\n  <body>\n    <main class=\"container\">\n      {}\n </main>\n  <script src=\"js/jquery.min.js\"></script>\n    <script src=\"js/bootstrap.min.js\"></script>\n  </body>\n</html>\n",
    "header": "<div class=\"header clearfix\">\n  <nav>\n    <ul class=\"nav nav-pills pull-left\">\n      {}\n    </ul>\n  </nav>\n</div>\n",
    "btn-active": "<li class=\"active\"><a href=\"#\">[]</a></li>\n",
    "btn-inactive": "<li><a href=\"#\">[]</a></li>\n",
    "row": "<div class=\"row\">{}</div>\n",
    "single": "<div class=\"col-lg-12\">\n{}\n</div>\n",
    "double": "<div class=\"col-lg-6\">\n{}\n</div>\n",
    "quadruple": "<div class=\"col-lg-3\">\n{}\n</div>\n",
    "btn-green": "<a class=\"btn btn-success\" href=\"#\" role=\"button\">[]</a>\n",
    "btn-orange": "<a class=\"btn btn-warning\" href=\"#\" role=\"button\">[]</a>\n",
    "btn-red": "<a class=\"btn btn-danger\" href=\"#\" role=\"button\">[]</a>",
    "big-title": "<h2>[]</h2>",
    "small-title": "<h4>[]</h4>",
    "text": "<p>[]</p>\n"
}

class Runner:

    def __init__(self):
        
        self.dsl_mapping = DEFAULT_DSL_MAPPING
        self.tokenizer = Tokenizer(filters='', split=" ", lower=False)
        self.tokenizer.fit_on_texts([VOCAB])
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(self.tokenizer.word_index)
        
        json_file = open("./save_model_bak/model_json.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./save_model_bak/weights.h5")
        
        self.model = loaded_model
        
    def convert_image(self, png_path, original_gui_filepath, print_generated_output, get_sentence_bleu, get_html):
        
        generated_gui = self.generate_gui(png_path, print_generated_output=print_generated_output)

        if get_sentence_bleu:
            print("Accuracy score: " + str(self.get_accuracy(original_gui_filepath, generated_gui)))

        if get_html:
            self.generate_html(generated_gui)

    def generate_gui(self, png_path, print_generated_output):
        photo = self.get_img_features(png_path)
        gui_encoding = '<START> '
        iteration = 0
        while(iteration < 150):
            all_sequence = self.tokenizer.texts_to_sequences([gui_encoding])
            sequence = all_sequence[0]
            padded_sequence = pad_sequences([sequence], maxlen=48)
            # print(padded_sequence)
            prediction = self.model.predict([photo, padded_sequence])
            # print(prediction)
            element = self.getElement(prediction)
            if element is None:
                break
            elif element == '<END>':
                gui_encoding += element + ' '
                break
            else:
                gui_encoding += element + ' '
            iteration += 1
        generated_gui = gui_encoding.split()
        if print_generated_output:
            print(generated_gui)
        return generated_gui


    def getElement(self, prediction):
        pKey = np.argmax(prediction)
        for element, key in self.tokenizer.word_index.items():
            if(key == pKey):
                return element
            else:
                pass
        return None

    def get_img_features(self, png_path):
        img_grey = cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2GRAY)
        img_shape = (256, 256, 3)
        # img_shape = (224, 224, 3)
        # 
        img_mapped = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 9)
        bg_img = np.ones(img_shape)
        img_layered = np.repeat(img_mapped[..., None], 3, axis=2)
        bg_img *= 255
        scaled_img = cv2.resize(img_layered, (200, 200), interpolation=cv2.INTER_AREA)
        bg_img[27:227, 27:227, :] = scaled_img
        return np.array([bg_img / 255])

    def get_accuracy(self, original_gui_filepath, generated_gui):
        generated_gui = self.balance_gui(generated_gui)
        hypothesis = generated_gui[1:-1]
        references = self.load_gui_doc(original_gui_filepath)
        return sentence_bleu([references], hypothesis)
    
    def load_gui_doc(self, gui_filepath):
        file = open(gui_filepath, 'r')
        gui = file.read()
        gui = ' '.join(gui.split()).replace(',', ' ,')
        file.close()
        return self.balance_gui(gui.split())

    def balance_gui(self, gui):
        btns_to_replace = ['btn-green', 'btn-red']
        
        normalized_gui = []
        normalized_btns = []

        for token in gui:
            if(token in btns_to_replace):
                normalized_btns.append('btn-orange')
            else:
                normalized_btns.append(token)
        
        for token in normalized_btns:
            if(token == 'btn-inactive'):
                normalized_gui.append('btn-active')
            else:
                normalized_gui.append(token)

        return normalized_gui

    def generate_html(self, generated_gui):
        generated_html = self.compile(generated_gui)
        output_filepath = "website/index.html"
        with open(output_filepath, 'w') as output_file:
            output_file.write(generated_html)
            print("Saved generated HTML to {}".format(output_filepath))

    def getCleanedGui(self, generate_gui):
        # Drop Start
        cleaned_gui = generate_gui[1: ]
        # Drop End
        cleaned_gui = cleaned_gui[ :-1]
        cleaned_gui = ' '.join(cleaned_gui)
        cleaned_gui = cleaned_gui.replace(' ', '').replace('}', '9999}9999').replace('{', '{9999')
        cleaned_gui = cleaned_gui.split('9999')
        cleaned_gui = list(filter(None, cleaned_gui))
        return cleaned_gui

    def compile(self, generated_gui):
        
        cleaned_gui = self.getCleanedGui(generated_gui)
        
        opening_tag = self.dsl_mapping["opening-tag"]
        closing_tag = self.dsl_mapping["closing-tag"]
        content_holder = opening_tag + closing_tag
        root = Node("body", None, content_holder)
        
        current_parent = root
        
        for token in cleaned_gui:
            
            token = token.replace(" ", "").replace("\n", "")
            
            if(token.find(opening_tag) != -1):
                token = token.replace(opening_tag, "")
                element = Node(token, current_parent, content_holder)
                current_parent.add_child(element)
                current_parent = element
            elif(token.find(closing_tag) != -1):
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = Node(t, current_parent, content_holder)
                    current_parent.add_child(element)

        output_html = root.render(self.dsl_mapping)

        if(output_html is None): 
            return "HTML Parsing Error"

        return output_html