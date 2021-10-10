import torch
model = torch.load('Drone detect Colab',  path='yolov5s.pt')  # local repo

# img = 'data\images\bus.jpg'

# Inference
# results = model(img)

# results.pandas().xyxy[0]