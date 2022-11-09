from flask import*
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid
import os

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
currentDir = os.path.dirname(__file__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app.config["UPLOAD_FOLDER"] = "static/"

@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/success', methods = ['GET', 'POST'])
def display_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        f.save(app.config['UPLOAD_FOLDER'] + filename)

        file = open(app.config['UPLOAD_FOLDER'] + filename,"rb")
        import requests
        import random

        n = random.choice([1,2,3,4,5,6,7,8,9])
        num = str(n)
        name = "Removed"+num+".png"
        path = app.config['UPLOAD_FOLDER'] + filename

        def save_output(image_name, output_name, pred, d_dir, type):
            predict = pred
            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()
            im = Image.fromarray(predict_np * 255).convert('RGB')
            image = io.imread(image_name)
            imo = im.resize((image.shape[1], image.shape[0]))
            pb_np = np.array(imo)
            if type == 'image':
                # Make and apply mask
                mask = pb_np[:, :, 0]
                mask = np.expand_dims(mask, axis=2)
                imo = np.concatenate((image, mask), axis=2)
                imo = Image.fromarray(imo, 'RGBA')

            imo.save(d_dir + output_name)

        # Remove Background From Image (Generate Mask, and Final Results)

        def removeBg(imagePath):
            inputs_dir = os.path.join(currentDir, 'static/inputs/')
            results_dir = os.path.join(currentDir, 'static/results/')
            masks_dir = os.path.join(currentDir, 'static/masks/')

            # convert string of image data to uint8
            with open(imagePath, "rb") as image:
                f = image.read()
                img = bytearray(f)

            nparr = np.frombuffer(img, np.uint8)

            if len(nparr) == 0:
                return '---Empty image---'

            # decode image
            try:
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                # build a response dict to send back to client
                return "---Empty image---"

            # save image to inputs
            unique_filename = str(uuid.uuid4())
            cv2.imwrite(inputs_dir + unique_filename + '.jpg', img)

            # processing
            image = transform.resize(img, (320, 320), mode='constant')

            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpImg = np.expand_dims(tmpImg, 0)
            image = torch.from_numpy(tmpImg)

            image = image.type(torch.FloatTensor)
            image = Variable(image)

            d1, d2, d3, d4, d5, d6, d7 = net(image)
            pred = d1[:, 0, :, :]
            ma = torch.max(pred)
            mi = torch.min(pred)
            dn = (pred - mi) / (ma - mi)
            pred = dn

            save_output(inputs_dir + unique_filename + '.jpg', name, pred, results_dir, 'image')
            save_output(inputs_dir + unique_filename + '.jpg', unique_filename +
                        '.png', pred, masks_dir, 'mask')
            return "---Success---"

        # ------- Load Trained Model --------
        print("---Loading Model---")
        model_name = 'u2net'
        model_dir = os.path.join(currentDir, 'saved_models',
                                 model_name, model_name + '.pth')
        net = U2NET(3, 1)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        # ------- Load Trained Model --------
        removeBg(path)
        print("---Removing Background...")

    return send_file('static/results/'+name,as_attachment=True)
if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
