
# Global Imports
import os
import glob
import shutil
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Project Imports
from my_utils import split_data, order_test_set
from deeplearning_models import streetsigns_model, create_generators


if __name__ == "__main__":
    
    # Change False to True when you have to split Data in train and val sets
    if False:

        pathToData = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\Train"
        pathToSaveTrain = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\training_data\train"
        pathToSaveVal = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\training_data\val"
        split_data(pathToData = pathToData , pathToSaveTrain = pathToSaveTrain, pathToSaveVal = pathToSaveVal)

    # Change False to True when you have to order test set and extract labels
    if False:

        pathToImages = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\Test"
        pathToCsv = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\Test.csv"
        order_test_set(pathToImages, pathToCsv)

    pathToTrain = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\training_data\train"
    pathToVal = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\training_data\val"
    pathToTest = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\Test"
    
    
    # Change to True/False according to requiment, changing values will activate training and testing code
    TRAIN = True
    TEST = False
    
    # Parameters
    batch_size = 64
    epochs = 10

    trainGenerator, valGenerator, testGenerator = create_generators(batch_size, pathToTrain, pathToVal, pathToTest)
    nbr_classes = trainGenerator.num_classes
    
    if TRAIN:
        pathToSaveModel = "./Models"

     # Create Model Checkpoints i.e Check for accuracy at each epoch and save the model when maximum validation accuracy is found
        ckpt_saver = ModelCheckpoint( 
            pathToSaveModel,
            monitor = 'val_accuracy',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )

    # if validation accuracy does not improve after 10 epochs stop training
        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=10 
        )

        model = streetsigns_model(nbr_classes)

        # instead of passing a string an optimizer can be passed too
        # optimizer = tensorflow.keras.optimizers.Adam(
        #   learning_rate=0.001,
        #   beta_1=0.9,
        #   beta_2=0.999,
        #   epsilon=1e-07,
        #   amsgrad=False,
        #   name='Adam'
        # )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(
            trainGenerator,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = valGenerator,
            callbacks = [ckpt_saver,early_stop]
        )


    if TEST:
        # Load Model and show Model Summary, thereafter evaluate model on validation and test sets
        model = tensorflow.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set : ")
        model.evaluate(valGenerator)

        print("Evaluating test set : ")
        model.evaluate(testGenerator)

    

    

    

