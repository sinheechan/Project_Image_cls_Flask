# Image Classification Server
from flask import request
from flask import Flask
from flask import jsonify
import io
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import json

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# 이미 학습된 가중치를 사용하기 위해 `pretrained` 에 `True` 값
model = models.densenet121(pretrained=True)
# 모델을 추론에만 사용할 것이므로, `eval` 모드로
model.eval()

# 이미지 분류 json 파일 경로
imagenet_class_index = json.load(open('./imagenet_class_index.json'))
'''
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
'''
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, predicted_idx = outputs.max(1)
    predicted_idx = predicted_idx.item()
    predicted_class = imagenet_class_index[str(predicted_idx)]
    probability = outputs.softmax(dim=1)[0][predicted_idx].item()
    return predicted_idx, predicted_class, probability

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Image Classification Sample'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Request로부터 파일 받기
        file = request.files['file']

        # 파일을 바이트로
        img_bytes = file.read()

        # 예측해서 반환
        class_id, class_name, probability = get_prediction(image_bytes=img_bytes)
        probability = round(probability, 2)
        return jsonify({'class_id': class_id, 'class_name': class_name, 'probability': probability})


if __name__ == '__main__':
    app.run()