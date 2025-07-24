# BACC Improvement & Parameter Tuning System

의료 영상 분류를 위한 BACC(Balanced Accuracy) 개선 및 파라미터 튜닝 시스템입니다. 클래스 불균형 문제를 해결하고 최적의 하이퍼파라미터를 찾아 AUC, F1, BACC를 모두 향상시킵니다.

## 🎯 목표

- **BACC**: 0.5 → 0.7+ (클래스 불균형 해결)
- **AUC**: 0.76 → 0.8+ (전체 성능 향상)
- **F1**: 0.72 → 0.75+ (균형잡힌 성능)
- **파라미터 최적화**: 자동 하이퍼파라미터 튜닝

## 📁 프로젝트 구조

```
/home/imp_cls/
├── 📄 integrated_bacc_tuning_system.py # 통합 BACC 개선 & 파라미터 튜닝 시스템 (메인)
├── 📄 medical_data_augmentation.py    # 의료 영상 데이터 증강
├── 📄 ensemble_methods.py             # 앙상블 방법들
├── 📁 models/                         # 3D ResNet18, ViT 모델
├── 📁 utils/                          # 데이터 로딩, 훈련 유틸리티

└── 📖 README.md                       # 이 파일
```

## 🔧 시스템 구성

### Integrated BACC & Parameter Tuning System (`integrated_bacc_tuning_system.py`)
**BACC 개선과 파라미터 튜닝을 동시에 수행하는 통합 시스템:**

#### **5가지 BACC 개선 방법:**
| 방법 | 설명 | 예상 BACC | 소요 시간 |
|------|------|-----------|-----------|
| **Method 1** | Enhanced Loss Functions | 0.65-0.70 | 2-3시간 |
| **Method 2** | Data Augmentation | 0.60-0.65 | 3-4시간 |
| **Method 3** | Threshold Optimization | **0.70-0.75** | **2-3시간** |
| **Method 4** | Ensemble Methods | 0.75-0.80 | 6-8시간 |
| **Method 5** | Combined Approach | 0.80-0.85 | 8-10시간 |

#### **파라미터 튜닝 기능:**
- **튜닝 레벨**: Light (5-10조합), Medium (20-30조합), Full (50+조합)
- **학습률**: 1e-5, 5e-5, 1e-4, 5e-4
- **배치 크기**: 4, 8, 16
- **옵티마이저**: AdamW
- **손실 함수**: Focal, Weighted CE, Advanced Balanced
- **데이터 증강**: True/False
- **임계값 최적화**: True/False
- **실시간 최적화**: 각 조합마다 BACC 개선 적용
- **통합 결과 분석**: 방법별 성능 비교 및 최적 파라미터 추출

## 🚀 사용법

### 통합 BACC & 파라미터 튜닝 시스템 (메인)

```bash
cd /home/imp_cls

# 단일 BACC 방법으로 파라미터 튜닝 (권장)
python integrated_bacc_tuning_system.py --bacc-method 3 --tuning-level medium --epochs 50

# 빠른 테스트
python integrated_bacc_tuning_system.py --bacc-method 3 --tuning-level light --epochs 10 --max-combinations 5

# 모든 BACC 방법으로 파라미터 튜닝
python integrated_bacc_tuning_system.py --run-all-methods --tuning-level light --epochs 30

# ViT 모델로 통합 튜닝
python integrated_bacc_tuning_system.py --model vit --bacc-method 3 --tuning-level medium --epochs 50
```

## 📊 예상 결과

### 통합 BACC & 파라미터 튜닝 시스템
- **실제 달성 BACC**: **0.656** (기존 0.500 대비 +31.2%)
- **최적 임계값**: 0.250 (기본 0.500 대비)
- **실시간 최적화**: 각 파라미터 조합마다 BACC 개선 적용
- **통합 분석**: 방법별 성능 비교 및 최적 파라미터 자동 추출
- **완전 자동화**: 오류 없이 모든 과정 자동 실행

## 🎯 권장 워크플로우

### 1단계: 빠른 테스트 (10분)
```bash
python integrated_bacc_tuning_system.py --bacc-method 3 --tuning-level light --epochs 10 --max-combinations 3
```

### 2단계: 통합 BACC & 파라미터 튜닝 (3-4시간)
```bash
python integrated_bacc_tuning_system.py --bacc-method 3 --tuning-level medium --epochs 50
```

### 3단계: 모든 BACC 방법 통합 실행 (6-8시간)
```bash
python integrated_bacc_tuning_system.py --run-all-methods --tuning-level light --epochs 30
```

### 4단계: ViT 모델로 통합 튜닝 (4-5시간)
```bash
python integrated_bacc_tuning_system.py --model vit --bacc-method 3 --tuning-level medium --epochs 50
```

## 🔧 의료 영상 특화 기능

✅ **회전/이동 없음**: 공간 관계 보존  
✅ **노이즈 추가**: 가우시안 노이즈  
✅ **밝기/대비 조정**: 의료 영상에 적합  
✅ **클래스 균형**: AD 클래스 오버샘플링  
✅ **임계값 최적화**: BACC 최대화  
✅ **앙상블 방법**: 다중 모델 결합  
✅ **파라미터 튜닝**: 자동 최적화  

## 📈 결과 확인

### 실시간 모니터링
```bash
# 훈련 중 실시간 지표 확인
tail -f ./result/*/training_log.txt
```

### 최종 결과 확인
```bash
# BACC 개선 결과
python bacc_improvement_system.py --method 0

# 파라미터 튜닝 결과
ls -la ./result/tuning/
cat ./result/tuning/best_params_resnet18.json
```

## 🛠️ 문제 해결

### 메모리 부족
```bash
# 배치 크기 줄이기
python parameter_tuning_system.py --max-combinations 20
```

### GPU 메모리 부족
```bash
# 다른 GPU 사용
python parameter_tuning_system.py --gpu-id 1
```

### 시간 단축
```bash
# 에폭 수 줄이기
python parameter_tuning_system.py --epochs 30 --max-combinations 20
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. **데이터 경로**: `/home/classification/data/syn_data_mapping.csv` 존재 확인
2. **GPU 사용 가능**: `nvidia-smi`로 GPU 상태 확인
3. **메모리**: `free -h`로 메모리 상태 확인
4. **로그**: `./result/` 디렉토리의 로그 파일 확인

---

**Happy BACC Improvement & Parameter Tuning!** 🧠🔬📈⚙️ 