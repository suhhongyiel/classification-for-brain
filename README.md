# BACC Improvement System for Medical Image Classification

의료 영상 분류를 위한 BACC(Balanced Accuracy) 개선 시스템입니다. 클래스 불균형 문제를 해결하여 AUC, F1, BACC를 모두 향상시키는 다양한 방법을 제공합니다.

## 🎯 목표

- **BACC**: 0.5 → 0.7+ (클래스 불균형 해결)
- **AUC**: 0.76 → 0.8+ (전체 성능 향상)
- **F1**: 0.72 → 0.75+ (균형잡힌 성능)

## 📁 프로젝트 구조

```
/home/imp_cls/
├── data/
│   └── syn_data_mapping.csv          # 원본 데이터 매핑
├── models/                           # 모델 정의
├── utils/                            # 유틸리티 함수들
├── medical_data_augmentation.py      # 의료 영상 데이터 증강
├── ensemble_methods.py               # 앙상블 방법들
├── bacc_improvement_system.py        # 메인 BACC 개선 시스템
└── README.md                         # 이 파일
```

## 🔧 개선 방법들

### 1. Enhanced Loss Functions (고급 손실 함수)
- **Focal Loss**: 어려운 샘플에 집중
- **Label Smoothing**: 일반화 성능 향상
- **KL Divergence**: 추가 정규화

### 2. Data Augmentation (데이터 증강)
- **의료 영상 특화**: 회전/이동 없음
- **노이즈 추가**: 가우시안 노이즈
- **밝기/대비 조정**: 의료 영상에 적합
- **클래스 균형**: AD 클래스 오버샘플링

### 3. Threshold Optimization (임계값 최적화)
- **BACC 최적화**: 0.1~0.9 범위에서 최적 임계값 탐색
- **자동 최적화**: 검증 세트에서 최적 임계값 찾기

### 4. Ensemble Methods (앙상블 방법)
- **다중 모델**: 서로 다른 설정으로 3개 모델 훈련
- **앙상블 예측**: 가중 평균으로 예측 결합
- **Threshold 최적화**: 앙상블 결과에 대한 임계값 최적화

### 5. Combined Approach (통합 접근법)
- **모든 방법 결합**: 최고 성능 방법들의 조합
- **시너지 효과**: 개별 방법보다 우수한 성능

## 🚀 사용법

### 기본 사용법

```bash
cd /home/imp_cls

# 모든 방법 실행 (권장)
python bacc_improvement_system.py --method 0 --model resnet18 --epochs 100 --gpu-id 0

# 특정 방법만 실행
python bacc_improvement_system.py --method 1 --model resnet18 --epochs 100 --gpu-id 0
```

### 개별 방법 실행

```bash
# 1. Enhanced Loss Functions
python bacc_improvement_system.py --method 1 --model resnet18 --epochs 100

# 2. Data Augmentation
python bacc_improvement_system.py --method 2 --model resnet18 --epochs 100

# 3. Threshold Optimization
python bacc_improvement_system.py --method 3 --model resnet18 --epochs 100

# 4. Ensemble Methods
python bacc_improvement_system.py --method 4 --model resnet18 --epochs 100

# 5. Combined Approach
python bacc_improvement_system.py --method 5 --model resnet18 --epochs 100
```

### 모델 선택

```bash
# ResNet18 (권장)
python bacc_improvement_system.py --method 0 --model resnet18

# Vision Transformer
python bacc_improvement_system.py --method 0 --model vit
```

### GPU 설정

```bash
# GPU 0 사용
python bacc_improvement_system.py --method 0 --gpu-id 0

# GPU 1 사용
python bacc_improvement_system.py --method 0 --gpu-id 1
```

## 📊 예상 결과

| 방법 | 예상 BACC | 예상 AUC | 예상 F1 | 소요 시간 |
|------|-----------|----------|---------|-----------|
| **Enhanced Loss** | 0.65-0.70 | 0.78-0.82 | 0.73-0.77 | 2-3시간 |
| **Data Augmentation** | 0.60-0.65 | 0.75-0.80 | 0.70-0.75 | 3-4시간 |
| **Threshold Opt** | 0.70-0.75 | 0.76-0.80 | 0.72-0.76 | 2-3시간 |
| **Ensemble** | 0.75-0.80 | 0.80-0.85 | 0.75-0.80 | 6-8시간 |
| **Combined** | 0.80-0.85 | 0.82-0.87 | 0.77-0.82 | 8-10시간 |

## 🔍 결과 확인

### 실시간 모니터링
```bash
# 훈련 중 실시간 지표 확인
tail -f ./result/method*/training_log.txt
```

### 최종 결과 확인
```bash
# 모든 방법의 결과 요약
python bacc_improvement_system.py --method 0
```

### 상세 결과 분석
```bash
# 각 방법별 상세 결과
ls -la ./result/method*/
cat ./result/method*/final_results.json
```

## ⚙️ 고급 설정

### 데이터 증강 강도 조정
```python
# medical_data_augmentation.py에서 조정
noise_prob = 0.3      # 노이즈 확률
brightness_prob = 0.3 # 밝기 조정 확률
contrast_prob = 0.3   # 대비 조정 확률
```

### 앙상블 가중치 조정
```python
# ensemble_methods.py에서 조정
weights = [0.4, 0.3, 0.3]  # 모델별 가중치
```

### 임계값 탐색 범위 조정
```python
# bacc_improvement_system.py에서 조정
thresholds = np.arange(0.1, 0.9, 0.01)  # 탐색 범위
```

## 🛠️ 문제 해결

### 메모리 부족
```bash
# 배치 크기 줄이기
python bacc_improvement_system.py --method 0 --batch-size 4
```

### GPU 메모리 부족
```bash
# 다른 GPU 사용
python bacc_improvement_system.py --method 0 --gpu-id 1
```

### 훈련 중단 시 재시작
```bash
# 동일한 명령어로 재실행 (자동으로 이어서 실행)
python bacc_improvement_system.py --method 0
```

## 📈 성능 최적화 팁

1. **GPU 메모리**: 8GB 이상 권장
2. **배치 크기**: 8-16이 최적
3. **학습률**: 1e-5가 가장 안정적
4. **에폭**: 100 에폭으로 충분한 수렴
5. **Early Stopping**: 25 patience로 과적합 방지

## 🎯 권장 워크플로우

### 1단계: 빠른 테스트 (30분)
```bash
python bacc_improvement_system.py --method 3 --epochs 30
```

### 2단계: 개별 방법 테스트 (2-3시간)
```bash
python bacc_improvement_system.py --method 1 --epochs 100
python bacc_improvement_system.py --method 3 --epochs 100
```

### 3단계: 전체 시스템 실행 (8-10시간)
```bash
python bacc_improvement_system.py --method 0 --epochs 100
```

### 4단계: 최적 방법 재실행 (2-3시간)
```bash
# 최고 성능 방법만 재실행
python bacc_improvement_system.py --method 4 --epochs 100
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. **데이터 경로**: `/home/imp_cls/data/syn_data_mapping.csv` 존재 확인
2. **GPU 사용 가능**: `nvidia-smi`로 GPU 상태 확인
3. **메모리**: `free -h`로 메모리 상태 확인
4. **로그**: `./result/` 디렉토리의 로그 파일 확인

---

**Happy BACC Improvement!** 🧠🔬📈 