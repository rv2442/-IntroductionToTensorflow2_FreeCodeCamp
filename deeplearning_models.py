import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# tensorflow.keras.model (functional approach)
def functional_model():

    my_input = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs = my_input, outputs = x)

    return model


# tensorflow.keras.model (inherit from class)
class MyCustomModel(tensorflow.keras.Model):

    def __init__(self):
        super().__init__()

        
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')


    def call(self, my_input):

        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def  streetsigns_model(nbr_classes):

    my_input = Input(shape=(60, 60, 3))

    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs = my_input, outputs = x)


def create_generators(batch_size, trainDataPath, valDataPath, testDataPath):

    #use different preprocessors for training and testing if you design a custom one for training
    trainPreProcessor = ImageDataGenerator(
        rescale = 1/255. ,
        rotation_range = 10,
        width_shift_range = 0.1 # if used in preprocessor dont use in generators
    )

    testPreProcessor = ImageDataGenerator(
        rescale = 1/255.
    )
    
    trainGenerator = trainPreProcessor.flow_from_directory(
        trainDataPath, 
        class_mode = 'categorical',
        target_size = (60,60),
        color_mode = 'rgb',
        shuffle = True,
        batch_size = batch_size
    )

    valGenerator = testPreProcessor.flow_from_directory(
        valDataPath, 
        class_mode = 'categorical',
        target_size = (60,60),
        color_mode = 'rgb',
        shuffle = False,
        batch_size = batch_size
    )

    testGenerator = testPreProcessor.flow_from_directory(
        testDataPath, 
        class_mode = 'categorical',
        target_size = (60,60),
        color_mode = 'rgb',
        shuffle = False,
        batch_size = batch_size
    )

    return trainGenerator, valGenerator, testGenerator


# will only run when this file is executed as main, works like a code used for testing purposes
# will not activate when other files are executed as main
if __name__ == "__main__":

    model = streetsigns_model(10)
    model.summary()
