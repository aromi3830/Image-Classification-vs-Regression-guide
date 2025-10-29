# Image-Classification-vs-Regression-guide
# Image Classification vs Regression 비교

머신러닝에서 가장 기본이 되는 두 가지 문제 유형인 Classification(분류)과 Regression(회귀)의 차이를 정리합니다.

## 📊 핵심 차이점 비교표

| 구분 | Image Classification | Regression |
|------|---------------------|------------|
| **출력 유형** | 카테고리/클래스 (이산적) | 연속적인 숫자 값 |
| **목적** | 이미지가 어떤 범주에 속하는지 판별 | 이미지로부터 수치를 예측 |
| **출력 예시** | "고양이", "개", "자동차" | 25.7, 180.5, 0.85 |
| **정답 형태** | 라벨(Label) | 실수 값(Real Number) |
| **평가 지표** | 정확도(Accuracy), F1-Score | MSE, RMSE, MAE, R² |

---

## 🎯 Image Classification (이미지 분류)

### 정의
이미지를 **미리 정의된 카테고리 중 하나로 분류**하는 작업

### 특징
- **출력**: 카테고리/클래스
- **예측값**: 확률 분포 (각 클래스에 속할 확률)
- **손실 함수**: Cross-Entropy Loss
- **활성화 함수**: Softmax (다중 분류), Sigmoid (이진 분류)

### 실제 예시

| 입력 이미지 | 예측 결과 | 출력 형태 |
|------------|----------|----------|
| 고양이 사진 | "고양이" | 클래스 라벨 |
| X-ray 이미지 | "정상" / "비정상" | 이진 분류 |
| 손글씨 숫자 | 0, 1, 2, ..., 9 | 10개 클래스 |
| 얼굴 사진 | "기쁨", "슬픔", "화남" | 감정 카테고리 |

### 예측 출력 예시
```python
# Softmax 출력 (확률 분포)
{
  "고양이": 0.85,
  "개": 0.10,
  "토끼": 0.05
}
# 최종 예측: "고양이" (가장 높은 확률)
```

### 주요 활용 분야
- 🖼️ 물체 인식 (ImageNet)
- 🏥 의료 영상 진단 (암/정상)
- 😊 얼굴 감정 인식
- 🚗 자율주행 (신호등, 표지판 인식)
- 📧 스팸 메일 분류

---

## 📈 Regression (회귀)

### 정의
이미지로부터 **연속적인 숫자 값을 예측**하는 작업

### 특징
- **출력**: 실수 값 (Real Number)
- **예측값**: 연속적인 숫자
- **손실 함수**: MSE (Mean Squared Error), MAE
- **활성화 함수**: Linear (선형) 또는 없음

### 실제 예시

| 입력 이미지 | 예측 결과 | 출력 형태 |
|------------|----------|----------|
| 사람 얼굴 | 25.7세 | 나이 (연속값) |
| 부동산 사진 | 3억 5천만원 | 가격 |
| 위성 이미지 | 32.5°C | 온도 |
| CT 스캔 | 종양 크기: 2.3cm | 크기 측정 |

### 예측 출력 예시
```python
# 직접적인 숫자 값
{
  "예측 나이": 25.7,
  "신뢰 구간": [23.5, 27.9]
}
```

### 주요 활용 분야
- 👤 나이 예측 (얼굴 사진)
- 💰 부동산 가격 예측
- 📏 객체 크기/거리 측정
- 🌡️ 온도/습도 예측
- 💊 약물 농도 예측

---

## 🔄 상세 비교표

### 1. 출력 특성

| 항목 | Classification | Regression |
|------|---------------|-----------|
| 출력 범위 | 제한적 (클래스 개수만큼) | 무한대 (연속적) |
| 출력 형태 | "A", "B", "C" | 1.5, 2.7, 100.3 |
| 중간값 | 의미 없음 | 의미 있음 |
| 순서 | 일반적으로 없음 | 있음 (크기 비교 가능) |

### 2. 네트워크 구조

| 항목 | Classification | Regression |
|------|---------------|-----------|
| 출력층 뉴런 수 | 클래스 개수 | 1개 (단일 값) |
| 출력 활성화 함수 | Softmax, Sigmoid | Linear, ReLU |
| 손실 함수 | Cross-Entropy | MSE, MAE, Huber |

### 3. 평가 지표

| Classification | Regression |
|---------------|-----------|
| Accuracy (정확도) | MSE (평균 제곱 오차) |
| Precision (정밀도) | RMSE (제곱근 평균 제곱 오차) |
| Recall (재현율) | MAE (평균 절대 오차) |
| F1-Score | R² Score (결정 계수) |
| Confusion Matrix | Residual Plot |

---

## 💻 코드 비교

### Image Classification 예시
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 분류 모델 (10개 클래스)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10개 클래스
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # 분류용 손실 함수
    metrics=['accuracy']
)

# 예측
prediction = model.predict(image)
# 출력: [0.1, 0.05, 0.8, 0.02, ...] (확률 분포)
class_id = np.argmax(prediction)  # 가장 높은 확률의 클래스
```

### Regression 예시
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 회귀 모델 (나이 예측)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # 1개 출력 (나이)
])

model.compile(
    optimizer='adam',
    loss='mse',  # 회귀용 손실 함수
    metrics=['mae']
)

# 예측
prediction = model.predict(image)
# 출력: [25.7] (실제 나이 값)
```

---

## 🎨 시각적 비교 다이어그램

### Classification 흐름도
```
입력 이미지
    ↓
CNN 특징 추출
    ↓
Fully Connected Layer
    ↓
Softmax 활성화
    ↓
확률 분포 [0.1, 0.8, 0.1]
    ↓
최대값 선택 (argmax)
    ↓
클래스 라벨: "고양이"
```

### Regression 흐름도
```
입력 이미지
    ↓
CNN 특징 추출
    ↓
Fully Connected Layer
    ↓
Linear 출력
    ↓
연속 값: 25.7
    ↓
나이: 25.7세
```

---

## 🔀 하이브리드: Multi-Task Learning

하나의 모델에서 Classification과 Regression을 **동시에** 수행할 수 있습니다.

| 작업 | 예시 |
|------|------|
| **Classification** | 얼굴 → 성별 (남/여) |
| **Regression** | 얼굴 → 나이 (25.7세) |

### 하이브리드 모델 구조
```python
# 멀티 태스크 모델
inputs = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = layers.Flatten()(x)

# Classification 출력
classification_output = layers.Dense(2, activation='softmax', name='gender')(x)

# Regression 출력
regression_output = layers.Dense(1, name='age')(x)

model = models.Model(inputs=inputs, 
                    outputs=[classification_output, regression_output])
```

---

## 📚 실제 사례 비교표

| 프로젝트 | 작업 유형 | 입력 | 출력 |
|---------|---------|------|------|
| **MNIST** | Classification | 손글씨 숫자 | 0~9 (10개 클래스) |
| **ImageNet** | Classification | 일반 이미지 | 1000개 객체 클래스 |
| **얼굴 나이 예측** | Regression | 얼굴 사진 | 나이 (0~100세) |
| **자율주행 조향각** | Regression | 도로 이미지 | 조향각 (-30° ~ 30°) |
| **암 진단** | Classification | X-ray | 악성/양성 |
| **종양 크기** | Regression | CT 스캔 | 크기 (cm) |

---

## ❓ 선택 가이드

### Classification을 사용해야 하는 경우
✅ 결과가 **카테고리**로 구분될 때
✅ "어떤 종류인가?"에 답할 때
✅ 예: 이것은 고양이인가 개인가?

### Regression을 사용해야 하는 경우
✅ 결과가 **숫자 값**일 때
✅ "얼마나?"에 답할 때
✅ 예: 이 사람의 나이는?

### 둘 다 사용할 수 있는 경우
⚠️ 문제를 재정의할 수 있는 경우
- 나이 예측: Regression (25.7세) 또는 Classification (20대, 30대...)
- 온도 예측: Regression (25.3°C) 또는 Classification (춥다, 따뜻하다, 덥다)

---

## 🎓 핵심 정리

| | Classification | Regression |
|---|---------------|-----------|
| **한 줄 요약** | "무엇인가?" | "얼마인가?" |
| **출력** | 라벨 | 숫자 |
| **목표** | 분류 정확도 최대화 | 예측 오차 최소화 |
| **예시** | 강아지/고양이 구분 | 나이 예측 |

---

## 📖 참고 자료

- [Coursera - Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Kaggle Image Classification Competitions](https://www.kaggle.com/competitions)

---

## 💡 추가 학습 키워드

### Classification 관련
- Softmax Function
- Cross-Entropy Loss
- One-Hot Encoding
- Confusion Matrix
- ROC Curve, AUC

### Regression 관련
- Mean Squared Error (MSE)
- Gradient Descent
- Overfitting/Underfitting
- Regularization (L1, L2)
- Residual Analysis

---

⭐ 이 문서가 도움이 되셨다면 GitHub Star를 눌러주세요!
