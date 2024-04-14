# Project_Image_cls_Flask

Flask 서버 구축 및 torchvision 이미지 분류 모델 별 성능 Test
<br /><br /> 
<img src="image/dog2.jpg">
<img src="image/Flask_server_result.png">

<br /><br /> 
## Object

Flask는 웹 애플리케이션을 구축하는 데 필요한 핵심 기능을 제공합니다.

해당 모델에서는 Mlops를 사용 하기 전 기초적인 Flask 서버를 구축하는 형태를 학습하고

이미지 분류 모델서빙을 진행하면서 torchvision에서 제공하는 분류모델의 성능과 특징 비교하는 것을 목적으로 모델을 생성합니다.
<br /><br /> 
## torchvision_cls_model_probability

모델 별 분류 카테고리의 결과는 아래와 같습니다.

|Cls_Model|cat|dog1|dog2|
|------|---|---|---|
|densenet121|0.49|Failure|0.79|
|resnet18|0.69|0.21|0.79|
|mobilenet_v2|0.60|0.49|0.63|
|vgg16|0.56|0.69|0.89|

<br /><br /> 
## Libraries used / Version

- numpy 1.26.4
- Flask  2.3.0
- flask-restx 1.3.0
- torchvision 0.17.1
- jsonify

<br /><br /> 
## Result

총 4개의 분류 모델에 대한 정확도를 테스트 한 결과, input 데이터 별 모델의 성능이 상이하여 특정 모델에 대한 우선순의 판별이 불가능하며

결국 이미지 데이터 분류 시 input 데이터의 특성과 모델이 선행 모델 자료와의 연관성을 근거로 모델을 선정하는 것이 최선의 성능이 나올 것으로 판단됩니다.
