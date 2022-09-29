import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import numpy as np

def split_data(pathToData, pathToSaveTrain, pathToSaveVal, splitSize = 0.1):
    
    folders = os.listdir(pathToData)

    for folder in folders:
        
        fullPath = os.path.join(pathToData, folder)
        imagesPaths = glob.glob(os.path.join(fullPath,'*.png'))

        x_train, x_val = train_test_split(imagesPaths, test_size = splitSize)

        for x in x_train:
            
            # basename = os.path.basename(x)
            pathToFolder = os.path.join(pathToSaveTrain, folder)

            if not os.path.isdir(pathToFolder):
                os.makedirs(pathToFolder)

            shutil.copy(x, pathToFolder)

        for x in x_val:
            
            # basename = os.path.basename(x)
            pathToFolder = os.path.join(pathToSaveVal, folder)

            if not os.path.isdir(pathToFolder):
                os.makedirs(pathToFolder)

            shutil.copy(x, pathToFolder)



def order_test_set(pathToImages, pathToCsv):
    
    testset = {}

    try:
        with open(pathToCsv, 'r') as csvFile:

            reader = csv.reader(csvFile, delimiter=',')
            
            for i, row in enumerate(reader):

                if i == 0:
                    continue
                
                img_name = row[-1].replace('Test/','')
                label = row[-2]

                pathToFolder = os.path.join(pathToImages, label)

                if not os.path.isdir(pathToFolder):
                    os.makedirs(pathToFolder)

                imgFullPath = os.path.join(pathToImages, img_name)
                shutil.move(imgFullPath, pathToFolder)

    except:
        print('[INFO] : Error reading csv file')
