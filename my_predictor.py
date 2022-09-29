import tensorflow as tf
import numpy as np


def predict_with_model(model, imgPath):
    image = tf.io.read_file(imgPath)
    image = tf.image.decode_png(image, channels = 3)
    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    image = tf.image.resize(image, [60,60]) # (60,60,3)
    image = tf.expand_dims(image, axis = 0) # (1,60,60,3)

    predictions = model.predict(image) # Probabilities of labeled classes
    predictions = np.argmax(predictions)

    return predictions


if __name__ == "__main__":
    imgPath = r"C:\Users\Rahul\OneDriveSky\Desktop\PROJ FILES\ADAS\archive\Train\34\00034_00000_00023.png"
    model = tf.keras.models.load_model('./Models')

    prediction = predict_with_model(model, imgPath)
    print(f"prediction = {prediction}")