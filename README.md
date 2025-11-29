# dynamic_quantization_analysis

# 📊 Dynamic Quantization Overhead Analysis

> **Quantifying the Cost of AbsMax Operations in Dynamic Quantization on GPU**

이 저장소는 PyTorch를 사용하여 GPU 상에서 **Dynamic Quantization(동적 양자화)**를 수행할 때, 스케일링 팩터(Scaling Factor) 계산을 위한 핵심 연산인 **AbsMax (Absolute Maximum)**가 전체 추론 대기 시간(Latency)에서 차지하는 비중을 분석한 벤치마크 프로젝트입니다.

## 🎯 Project Objective

Dynamic Quantization은 런타임에 입력 텐서의 활성화 값(Activation) 범위를 계산하여 `int8`로 변환합니다. 이 과정에서 `max(abs(tensor))` 연산이 필수적으로 수행됩니다.

이 프로젝트는 다양한 **Quantization Granularity(양자화 단위)**에 따라 AbsMax 연산의 오버헤드를 측정하고, 입력 텐서의 크기 변화에 따른 성능 특성을 시각화합니다.

## 🧪 Methodology

* **Timing**: `torch.cuda.Event`를 사용하여 마이크로초($\mu s$) 단위의 정밀한 커널 실행 시간을 측정합니다.
* **Profiling**: `nvtx`를 사용하여 프로파일링 툴(Nsight Systems 등)에서 식별 가능한 구간을 태깅합니다.
* **Granularity Types**:
    1.  **Per-Tensor**: 텐서 전체에서 하나의 Max 값을 계산.
    2.  **Per-Token**: 각 토큰(마지막 차원) 별로 Max 값을 계산.
    3.  **Per-Channel**: 채널 단위로 Max 값을 계산.

## 🛠️ Tech Stack

| Category | Technology |
| :--- | :--- |
| **Framework** | PyTorch (CUDA Support) |
| **Profiling** | NVTX, torch.cuda.Event |
| **Data Analysis** | NumPy |
| **Visualization** | Matplotlib, Seaborn |

## 📊 Experiments & Visualization

노트북(`experiments.ipynb`)은 입력 텐서 `(Batch, Sequence, Hidden)`의 크기를 조절하며 벤치마크를 수행합니다.

### 1. 실험 변수
* **Sequence Length ($C$) 변화**: $B=10, D=2048$ 고정, $C \in [2000, 4000, 8000]$
* **Batch Size ($B$) 변화**: $C=4000, D=2048$ 고정, $B \in [20, 40, 80]$

### 2. 시각화 결과
벤치마크 결과는 **Stacked Bar Chart**로 시각화됩니다.
* **파란색/초록색 구간 (Absmax)**: 최대 절댓값 계산에 소요된 시간.
* **회색 구간 (Other Ops)**: 나눗셈, 반올림, 클램핑 등 나머지 양자화 연산 시간.
* 그래프 상단에는 전체 시간 대비 AbsMax 연산이 차지하는 **백분율(%)**이 표시됩니다.

<img src="https://via.placeholder.com/800x400.png?text=Sample+Visualization+Placeholder" alt="Visualization Example" width="800"/>
*(Note: 실제 그래프는 노트북 실행 시 생성됩니다.)*

## 🚀 How to Run

### 1. 환경 설정
NVIDIA GPU가 장착된 환경에서 다음 라이브러리를 설치합니다.

```bash
pip install torch torchvision numpy matplotlib seaborn nvtx
