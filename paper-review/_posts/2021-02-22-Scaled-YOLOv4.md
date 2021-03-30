---
layout: post
title: Scaled-YOLOv4
date: 2021-02-22
comments: true
categories: [paper-review]
tags: [detector]
---

> We show that the YOLOv4 object detection neural network based on the CSP approach, **scales both up and down and is applicable to small and large networks while maintaining optimal speed and accuracy. We propose a network scaling approach that modifies not only the depth, width, resolution, but also structure of the network.** YOLOv4-large model achieves state-of-the-art results: 55.5% AP (73.4% AP50) for the MS COCO dataset at a speed of∼16 FPS on Tesla V100, while with the test time augmenta- tion, YOLOv4-large achieves 56.0% AP (73.3 AP50). To the best of our knowledge, this is currently the highest ac- curacy on the COCO dataset among any published work. The YOLOv4-tiny model achieves 22.0% AP (42.0% AP50) at a speed of∼443 FPS on RTX2080Ti, while by using TensorRT, batch size = 4 and FP16-precision the YOLOv4-tiny achieves 1774 FPS. - Abstract of the paper

- Table of Contents
{:toc .large-only}

## Summary
---
- 연산량과 메모리 대역폭의 균형을 체계적으로 조절하는 small model에 대한 model scaling 방법 제안
- 단순하지만 효과적인 large model에 대한 scaling 방법 제안
- 위의 scaling 기법을 기반으로 만든 YOLOv4-tiny, YOLOv4-large 모델 제안

## Real-time object detection
---
- Detector는 one-stage model과 two-stage model로 구분할 수 있다.
- 각 모델의 inference time은 다음과 같은 공식으로 나타낼 수 있다.<br> (one-stage : $$T_{one}=T_1st$$, two-stage : $$T_{two}=T_1st + mT_2nd$$)
  - $$m$$은 임계치 보다 confidence score가 높은 제안 영역의 개수(region proposals)
  - two-stage detector는 m의 개수에 따라 inference time이 변동되며 고정된 값이 아닌 반면, one-stage detector는 상수이다.
    따라서 실시간 객체 탐지기는 대부분 one-stage detector가 사용된다.

## Model scaling
---
- 기존의 model scaling 기법은 conv layer를 증감함으로서 모델의 depth를 조절하였다.
- FPN이 보편적으로 사용되고 난뒤에는 pyramid의 여러가지 feature map의 조합을 통해서도 scaling을 수행하였다.
- 최근에는 network architecture search(NAS)를 이용하여 강화학습 기반의 알고리즘을 통해 최적의 구조를 탐색하는 방법이 연구되었다.
- EfficientDet은 모델의 depth, width 그리고 입력 이미지의 해상도를 변수로한 compound scaling search를 통해 다양한 detector 구조를 설계하였다.
- 또한 RegNet은 6개의 파라미터(depth, initial width 등)를 기반으로 최적의 구조를 탐색한다.
- model scaling에 대한 다양한 연구가 행해졌지만, 적은 논문만이 각 파라미터의 상관관계에 대해 언급하였고 따라서 저자들은 이점에 집중하여 연구하였다.

### Challenge
> we found that CSPDarknet53, which is the backbone of YOLOv4, matches almost all optimal architecture features obtained by network architecture search technique. The depth of CSPDarknet53, bottleneck ratio, width growth ratio between stages are 65, 1, and 2, respectively. Therefore, we developed model scaling technique based on YOLOv4 and proposed scaled-YOLOv4.

- YOLOv4의 backbone network인 CSPDarknet53은 네트워크 구조 검색 기법에 의해 얻어진 최적의 구조와 동일함을 발견하였다.
  그렇다면, YOLOv4를 기반으로한 model scaling 기법과 기법을 통해 새로운 구조를 만들 수 있을까?

## Principles of model scaling
---
<figure>
  <img src="https://drive.google.com/uc?id=1N7FRrxw2xbY_6F2Y02h8swkkz885PjUq" style="width:100%">
  <figcaption> k : CNN의 레이어 수, b : 각 base 레이어의 채널 수, w,h : feature map의 width, height를 의미한다. </figcaption>
</figure>
- 일반적인 CNN 모델에 대해 (1) 이미지 크기, (2) 레이어의 개수, (3) 채널의 개수 를 변화시킬 때, 정량적 비용이 어떻게 변하는지를 설명한다.
  CNN 모델로는 ResNet, ResNext, Darknet을 선정하였다.
- 이미지 크기, 레이어 개수, 채널 개수를 조정하는 하이퍼 파라미터로 각각 $$\alpha, \beta, \gamma$$로 표시한다.
- Table 1은 일반적인 CNN 모델에서 각 파라미터를 변화할 때 FLOPs가 어떻게 변하는지를 나타낸다. Table 2는 각 CNN 모델의 레이어를 CSP 구조로 바꾸어 FLOPs를 계산한 표이다.
- CSP 구조로 변경하면, 각 모델의 FLOP는 23.5%, 46.7%, 50.0%로 감소한다. <br>
  <span style="color:#DF8B00">
  → Res layer의 경우를 예로들면, $$\frac{whb^2(3/4+13k/16)}{17whkb^2/16} = \frac{3/4+13k/16}{17k/16} \approx \frac{13k/16}{17k/16} = 0.764$$로
    <br>$$k$$가 큰 값이고 상수항을 무시할 수 있기 때문에, 기존 대비 23.5% 감소하는 수치로 설명.
  </span>

### Scaling Large Models for High-End GPUs
- 앞서 살펴본 다양한 factor들의 최적 조합을 찾아야한다. 
- Classification과 다른 점은 detection은 성능이 receptive field의 크기에 영향을 받는다는 것이다. CNN 구조상 receptive field와 가장 큰 영향이 있는 것은 stage이다. 또한 FPN구조에서 stage가 높을 수록 큰 객체를 잘 탐지하기 적합하다는 사실이 알려져있다.
- Receptive field와 연관성이 있는 **입력 이미지의 크기, 모델의 width, depth, #stage**로 총 4개의 factor를 두었다. 이 중 입력 이미지의 크기, #stage가 가장 영향력이 크기때문에 먼저 compound scaling을 진행하고, real-time 조건에 맞추어 depth, width를 조절한다.

## Scaled-YOLOv4
---
- 일반 GPU에 사용하는 YOLOv4 모델(general GPUs), 모바일 기기와 같은 곳에 사용하는 모델(low-end GPUs), high-end GPUs에 사용하는 모델을 디자인 하였다.

### CSP-ized YOLOv4
- YOLOv4는 실시간 객체 탐지 모델로 이를 더 개선하여 YOLOv4-CSP모델을 디자인 하였다.
- Backbone
  - CSPDarknet53에서 residual block의 down-sampling을 위한 cross-stage 단계를 배제한다.
  - 이로인해 각 CSPDarknet의 연산량은 $$whb^2(9/4+3/4+5k/2)$$로 감소한다.
  - CSPDarknet53의 각 stage는 1-2-8-8-4개의 레이어를 갖는다. (단, 첫번째 CSP stage는 원본 Darknet residual layer로 구성)
<figure>
  <img src="https://drive.google.com/uc?id=1pPlECgczerAwEUDvOF95wHpAyi7euIyv" style="width:70%">
</figure>
- Neck
  - YOLOv4의 PAN 구조 또한 CSP로 변경한다.
  - CSP로 변경된 구조는 Figure2와 같으며, 기존 연산량 대비 40% 감소한 수치를 보여준다.

### YOLOv4-tiny
- YOLOv4-tiny는 edge device와 같은 low-end GPU에 최적화된 모델이다.
- YOLOv4-tiny는 YOLOv3-tiny의 기본 디자인을 따른다.
- backbone으로 PCB 구조의 CSPOSANet을 사용한다.

### YOLOv4-large
<figure>
  <img src="https://drive.google.com/uc?id=1anQ2jec6lR-FrJpT2hreNV_BSCi7Qpoi" style="width:100%">
</figure>
- YOLOv4-large는 클라우드 GPU와 같은 곳에 사용하는 모델로, 주로 높은 정확도를 위한 모델이다.
- compound scaling을 통한 p5, p6, p7 모델이 있으며 모든 레이어가 CSP화 되었다.

## Experiments
---
- 학습 및 테스트 데이터는 MSCOCO2017을 사용
- ImageNet pretrained 모델을 사용하지 않았으며, 모든 Scaled-YOLOv4는 scratch부터 학습한다.
- YOLOv4-tiny는 600에폭, YOLOv4-CSP는 300에폭, YOLOv4-large는 300에폭 학습 후 데이터 증강기법을 적용하여 150에폭을 추가로 학습한다.

### Ablation study
<figure>
  <img src="https://drive.google.com/uc?id=1SfiaAdCpX1A2NcYPwWjLZnXjjcM1_jhJ" style="width:70%">
</figure>
<span style="color:#DF8B00">자세한 수치는 논문을 참고할 것, 본 글에서는 요약만 기록하였다.</span>
- CSP-ized model
  - CSP화된 모델을 사용하면, 대략 32%의 연산량 감소효과가 있으며 약간의 AP 성능 향상도 있다.
- YOLOv4-tiny
  - PCB 테크닉을 통해 유연한 모델을 디자인 할 수 있음을 보였다.
  - 선형적인 scaling 감소는 한계가 있다.
  - NVIDIA Jetson TX2에서 41FPS, 28.7%의 AP 수치를 보인다. (TensorRT를 사용할 경우 100FPS)
- YOLOv4-large
  - YOLOv4-P6의 AP는 54.5%로 EfficientDet-D7의 AP인 53.7%보다 성능이 높다. 그럼에도 불구하고 inference 속도는 3.7배 정도 빠르다.
  - TTA(Test-time augmentation)를 사용하면 AP가 상승한다.

## Paper Info
---
> Title : Scaled-YOLOv4: Scaling Cross Stage Partial Network arXiv:2011.08036v2

> Paper link : [https://arxiv.org/abs/2011.08036](https://arxiv.org/abs/2011.08036)

> Publised year : 22 Feb 2021

> Authors : Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao

---


<br>