# SmartDocument 
# By @ginel.d 
# prediction test
# make a prediction for a new digit image using mnist pretrained model

import os

# disable warnings about AVX CPU in tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# disable other commons warnings in console
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


'''
We do have 3 classes:
0: Contract 
1: transfert_form
2: withdrawral_form

'''


# load an image and predict the class
def run_example():
    # load the image
    img = load_image('img_test/80312639_996801677338489_5222169629043982336_n.png')
    # load model
    model = load_model('trained_digit.h5')
    # predict the class i-e the detected digit
    digit = model.predict_classes(img)

    print(" Document Type is: ", digit[0])


# entry point, run the example
run_example()
