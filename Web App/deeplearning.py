import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

model = tf.keras.models.load_model('./static/models/object_detection.h5')


# Create a Pipeline model
def object_detection(path, filename):

    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(224, 224))

    # Data Preprocessing
    image_arr_244 = img_to_array(image1)/255.0
    h, w, d = image.shape
    test_arr = image_arr_244.reshape(1, 224, 224, 3)

    # make predictions
    coords = model.predict(test_arr)

    # denormalize the predictions
    denorm = np.array([w, w, h, h])
    coords = coords*denorm
    coords = coords.astype(np.int32)

    # Draw the bounding on the top of the image
    xmin, xmax, ymin, ymax = coords[0]
    plt1 = (xmin, ymin)
    plt2 = (xmax, ymax)
    cv2.rectangle(image, plt1, plt2, (0, 255, 255), 3)

    # converting the image to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
    return coords


def OCR(path, filename):
    img = np.array(load_img(path))
    cods = object_detection(path, filename)
    xmin, xmax, ymin, ymax = cods[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)
    text = pt.image_to_string(roi)
    print(text)
    return text
