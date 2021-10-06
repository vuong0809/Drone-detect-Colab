import socketio
import json
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import  non_max_suppression,  set_logging
from utils.torch_utils import select_device, time_sync

@torch.no_grad()
def run(
        weights='yolov5s.pt',  # model.pt path(s)
        # weights='fire_model.pt',  # model.pt path(s)
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        ):

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    io = socketio.Client()
    io.connect('http://nguyentuanvuong.tk:8000')

    @io.on('StreamColab')
    def on_message(msg):
        results = {
            "socketID":"null",
            "img":"null",
            "results":[],
            "time":0
          }

        # results['img'] = msg['img']
        results["socketID"] = msg['socketID']

        imgText = msg['img'].encode('utf-8')
        imgText = base64.b64decode(imgText)
        image = np.asarray(bytearray(imgText), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        img = letterbox(image, 640, 32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        t1 = time_sync()
        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, max_det=max_det)
        t2 = time_sync()

        det = pred[0]

        s = ''

        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = f'{names[c]} {conf:.3f}'

            # x0 = float(torch.tensor(xyxy)[0].numpy())
            # y0 = float(torch.tensor(xyxy)[1].numpy())
            # x1 = float(torch.tensor(xyxy)[2].numpy())
            # y1 = float(torch.tensor(xyxy)[3].numpy())
            
            x0 = int(xyxy[0])
            y0 = int(xyxy[1])
            x1 = int(xyxy[2])
            y1 = int(xyxy[3])

            # results["results"].append({"x0":f'{x0:.3f}',"y0":f'{y0:.3f}',"x1":f'{x1:.3f}',"y1":f'{y1:.3f}',"name":names[c],"conf":f'{conf:.3f}',"label":label})
            results["results"].append({"x0":f'{x0}',"y0":f'{y0}',"x1":f'{x1}',"y1":f'{y1}',"name":names[c],"conf":f'{conf:.3f}',"label":label})
            
        results["output"] = f'{s}'
        results["time"] = f'{t2 - t1:.3f}'
        print(results["socketID"],results["output"], results["time"])
        io.emit('ResultsColab',json.dumps(results))


    @io.event()
    def connect_error(data):
        print("The connection failed!")
        sys.exit()

if __name__ == "__main__":
    run()
