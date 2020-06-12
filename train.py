from classes.DeStrap import *
from classes.Utils import *

def data_augmentation(training_path, validation_path):
    train_img_preprocessor = ImagePreprocessor()
    train_img_preprocessor.build_image_dataset(training_path)
    val_img_preprocessor = ImagePreprocessor()
    val_img_preprocessor.build_image_dataset(validation_path, augment_data=False)

def main():

    augment_data = False
    
    model = DeStrap()
    
    training_path, validation_path = "./all_data/training_set", "./all_data/validation_set"
    
    if augment_data:
        data_augmentation(training_path, validation_path)

    model.train(training_path=training_path, validation_path=validation_path, epochs=1)

if __name__ == "__main__":
    main()