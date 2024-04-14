# Local Test
import io
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

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
'''
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat
'''
import json
# 이미지 분류 json 파일 경로
imagenet_class_index = json.load(open('./imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

# 예측할 이미지 경로
with open("./cat.jpg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))