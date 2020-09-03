"""
This module is the main flask application.
"""

from flask import Flask, request, render_template
from blueprints import *
import cv2
import numpy as np
import tensorflow as tf
import re
import mahotas
import base64
import imutils

app = Flask(__name__)

model = tf.keras.models.load_model(
    '/Users/nhanpham/CoderSchool/AI_calculator/model/maths.h5')
label_names = ['0', '1', '2', '3', '4', '5',
               '6', '7', '8', '9' '10', '11', '12']

app.register_blueprint(home_page)


def parse_image(imgData):
    imgstr = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.decodebytes(imgstr)
    with open("output.jpg", "wb") as file:
        file.write(img_decode)
    return img_decode


def center_extent(image, eW, eH):
    """ Process contours into training size.
        Make sure the digit image is in the middle of the image.
        image: input image
        eW: new image width
        eH: new image height
    """

    # RESIZE
    # if horizontal size is longer
    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=eW-5)
    # if vertical size is longer
    else:
        image = imutils.resize(image, height=eH-5)

    # CENTERIZE
    # make a black canvas with train image size
    extent = np.zeros((eH, eW), dtype='uint8')

    # calculate the offsetX and offsetY of old image to new canvas
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2

    # make sure the digit image is in the middle of the canvas
    extent[offsetY:offsetY + image.shape[0],
           offsetX:offsetX+image.shape[1]] = image

    return extent


@app.route("/upload/", methods=["POST"])
def upload_file():
    """ Get the drawing numbers and return prediction 
    """
    img_raw = parse_image(request.get_data())
    nparr = np.fromstring(img_raw, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    (cnts, _) = cv2.findContours(edged.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0])
                   for c in cnts], key=lambda x: x[1])

    math_detect = []

    for (c, _) in cnts:
        #  find an approximate rectangle points (x,y) (x+w, y+h) around the binary image.
        (x, y, w, h) = cv2.boundingRect(c)

        # make sure the contours covering something big enough to be digits.
        if w >= 5 and h >= 5:
            single_digit_image = edged[y:y+h, x:x+w]

            # copy the digit to thresh to be preprocessed and be predicted later
            thresh = single_digit_image.copy()

            # resize and centerize the digit image
            thresh = center_extent(thresh, 28, 28)
            print(thresh.shape)

            # Normalize and expand dims so that image has the same shape with training image
            thresh = thresh / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = np.expand_dims(thresh, axis=0)

            predictions = model.predict(thresh)
            digit = np.argmax(predictions, axis=1)
            print(digit)
            if digit == "1":
                _count = 0
                mem = []
                for i in range(len(thresh[9])):
                    if thresh[9][i] > 0:
                        _count += 1
                        mem.append(i)
                if _count >= 3:
                    math_detect.append("1")
                else:
                    math_detect.append("/")
            else:
                math_detect.append(str(digit[0]))

    print('1111', math_detect[1])

    def convert_math(math_detect):
        """ Return + * and -, which were denoted as 10, 11, 12 during the training.
        """
        for i in range(0, len(math_detect)):
            if math_detect[i] == '10':
                math_detect[i] = '*'
            elif math_detect[i] == '11':
                math_detect[i] = '-'
            elif math_detect[i] == '12':
                math_detect[i] = '+'
        return math_detect

    def calculate_string(math_detect):
        """ Perform mathematics calculation 
        """
        math_detect = convert_math(math_detect)
        print('222', math_detect)
        calculator = ''.join(str(item) for item in math_detect)
        print('333', calculator)
        result = calculator
        return result

    result = calculate_string(math_detect)

    return result


@app.route("/calcu/", methods=["POST"])
def calcu():
    val = request.get_data()
    val = str(request.get_data())
    val1 = val[2:-1]

    result = str(eval(val1))
    return result


if __name__ == '__main__':
    app.run(debug=True)
