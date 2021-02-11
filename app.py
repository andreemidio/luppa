import json
import os

import jsonpickle
from flask import Flask, Response, request
import json
import os

import jsonpickle
from flask import Flask, Response, request

# Initialize the Flask application
from inference import inference

DEBUG = os.getenv("DEBUG")

app = Flask(__name__)


# route http posts to this method
@app.route('/api/predict', methods=['POST'])
def predict():
    image = json.loads(request.data)['image']

    inf = inference(data=image)

    response_pickled = jsonpickle.encode({"result": inf})

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    # start flask app
    app.run(host="0.0.0.0", port=5000, debug=DEBUG, threaded=True)
