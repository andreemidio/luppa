import base64
import os
from tempfile import gettempdir

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


global labels
model = load_model("model/luppa.h5")
labels = ["Soil", "Tree"]
img_name = None



def decode_image_from_bytes(data) -> str:
    """
    Executes string decode to image binary format.

    Parameters
    ----------
    data: str
        Image formatted in string
    """
    image = np.fromstring(data, np.uint8)
    image_cv = cv2.imdecode(image, cv2.IMREAD_COLOR)
    url2 = os.path.join(gettempdir(), 'image.jpg')
    cv2.imwrite(url2, image_cv)
    return url2


def get_image_from_base64(base64_data: str):
    """Convert base64 img to uint8.
    Parameters
    ----------
    base64_data: str
        Image formatted in base64.
    """
    nparr = base64.b64decode(base64_data)
    return decode_image_from_bytes(nparr)


def inference(data: str) -> str:
    """Run inference on image and return predictions.
    Parameters
    ----------
    data: str
        Image formatted in string
    """
    img = get_image_from_base64(data)
    img = cv2.imread(img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    result = np.argmax(result)
    return labels[result]


