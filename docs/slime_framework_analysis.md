# Slime 프레임워크 분석

## 분석 일자
2025-11-29

---

## 1. 프레임워크 개요

Slime은 THUDM에서 개발한 LLM RL 학습 프레임워크로, 3개의 모듈이 순환하는 구조입니다.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Training Module │ ←── │   Data Buffer   │ ←── │ Rollout Module  │
│ (Megatron/FSDP) │     │                 │     │ (SGLang+Router) │
└────────┬────────┘     └─────────────────┘     └────────▲────────┘
         │                                               │
         └───────────── Weight Sync ─────────────────────┘
```

---

## 2. 디렉토리 구조

```
slime/
├── slime/
│   ├── backends/              # 학습 백엔드
│   │   ├── megatron_utils/    # Megatron 구현
│   │   ├── fsdp_utils/        # PyTorch FSDP 구현
│   │   └── sglang_utils/      # SGLang 통합
│   ├── ray/                   # Ray 분산 오케스트레이션
│   │   ├── rollout.py         # RolloutManager
│   │   ├── train_actor.py     # TrainRayActor
│   │   └── rollout_data_source.py
│   ├── rollout/               # 데이터 생성
│   │   ├── sglang_rollout.py  # SGLang 기반 rollout
│   │   ├── rm_hub/            # Reward model 구현
│   │   └── filter_hub/        # Dynamic sampling filters
│   └── utils/
│       ├── types.py           # Sample 데이터 구조
│       ├── ppo_utils.py       # Advantage 계산
│       └── data.py            # Dataset/batching
├── train.py                   # 동기 학습 루프
├── train_async.py             # 비동기 학습 루프
└── examples/                  # 참조 구현 (tau-bench 포함)
```

---

## 3. 핵심 데이터 구조

### Sample 클래스

```python
@dataclass
class Sample:
    prompt: Union[str, list[dict]]  # 프롬프트 (멀티모달 가능)
    tokens: list[int]               # 토큰 ID (prompt + response)
    response: str                   # 텍스트 응답
    response_length: int            # 응답 토큰 길이
    reward: Union[float, dict]      # 스칼라 또는 구조화된 보상
    loss_mask: Optional[list[int]]  # 토큰별 loss mask
    status: Status                  # PENDING, COMPLETED, TRUNCATED, ABORTED
    rollout_log_probs: Optional[list[float]]
    metadata: dict                  # 커스텀 메타데이터
```

---

## 4. Advantage Estimator

### 지원 방식

| 방식 | 특징 | 적합한 경우 |
|-----|------|-----------|
| **GRPO** | Group Relative Policy Optimization | 같은 프롬프트에서 여러 샘플, Binary reward |
| GSPO | Context Parallel 지원 GRPO | 긴 시퀀스 |
| Reinforce++ | Token-level discounted returns | 중간 보상 있는 경우 |
| PPO | GAE 기반 | Critic 사용시 |

### GRPO Advantage 계산

```python
# 같은 프롬프트에서 n개 샘플
rewards = [1, 0, 0, 1]  # 4개 샘플
mean_reward = 0.5

# Advantage = reward - mean
advantages = [0.5, -0.5, -0.5, 0.5]

# STD normalization (선택)
if grpo_std_normalization:
    advantages = (advantages - mean) / std
```

---

## 5. Multi-turn 지원

### 토큰 누적 패턴

```python
sample.tokens = []  # 빈 상태로 시작

for turn in range(max_turns):
    # 1. 모델에서 action 생성
    action_tokens = model.generate(...)

    # 2. 토큰 누적
    sample.tokens.extend(action_tokens)
    sample.response += decode(action_tokens)

    # 3. 환경 step
    obs, reward, done, _ = env.step(decode(action_tokens))

    if done:
        break

# 최종 상태
sample.response_length = len(sample.tokens) - len(prompt_tokens)
sample.loss_mask = [1] * sample.response_length  # 또는 커스텀
sample.reward = final_reward
```

### Loss Mask 활용

```python
# Multi-turn with tool use
sample.loss_mask = [
    1, 1, 1,  # Turn 1: Agent response (학습)
    0, 0, 0,  # Tool response (마스킹)
    1, 1,     # Turn 2: Agent response (학습)
    0, 0,     # User response (마스킹)
    1, 1, 1,  # Turn 3: Agent response (학습)
]
```

---

## 6. τ²-Bench 통합 가이드

### 6.1 User Simulator 서빙 구조

외부 API 사용이 불가능한 환경에서는 내부 서버에 User Simulator 모델을 SGLang으로 서빙합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    학습 서버 (GPU 클러스터)                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Slime     │ ──▶ │  τ²-Bench   │ ──▶ │ User Sim    │   │
│  │ (Agent 학습) │     │    Env      │     │ API Call    │   │
│  └─────────────┘     └─────────────┘     └──────┬──────┘   │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                                                   ▼ HTTP (내부망)
┌─────────────────────────────────────────────────────────────┐
│                  내부 추론 서버 (별도 GPU)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SGLang Server (Qwen-7B-Instruct 등)                 │   │
│  │  http://internal-server:30001/v1/chat/completions   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**User Simulator 서버 실행 (내부 서버)**:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

SGLang은 OpenAI API 호환이므로 litellm으로 바로 호출 가능합니다.

### 6.2 Custom Generate Function

```python
# examples/tau2_bench/generate.py

async def generate(args, sample, sampling_params):
    """τ²-Bench 환경과 상호작용하여 trajectory 생성"""
    from tau2.gym.gym_agent import AgentGymEnv, register_gym_agent

    register_gym_agent()

    # 태스크 ID로 환경 초기화
    task_id = sample.prompt  # 또는 sample.metadata['task_id']
    env = AgentGymEnv(
        domain='telecom',
        task_id=task_id,
        max_steps=30,
        solo_mode=False,
        # 내부 SGLang 서버를 User Simulator로 사용
        user_llm='openai/qwen2.5-7b-instruct',  # litellm 포맷
        user_llm_args={
            'api_base': 'http://internal-server:30001/v1',
            'api_key': 'dummy',  # SGLang은 키 불필요
            'temperature': 0.0,
        },
    )

    obs, info = env.reset()
    sample.tokens = []
    sample.response = ""
    sample.loss_mask = []

    done = False
    while not done:
        # SGLang으로 action 생성
        action_text = await sglang_generate(
            prompt=build_prompt(obs, sample.response),
            sampling_params=sampling_params,
        )
        action_tokens = tokenize(action_text)

        # 토큰 누적 (agent response만 loss mask = 1)
        sample.tokens.extend(action_tokens)
        sample.loss_mask.extend([1] * len(action_tokens))
        sample.response += action_text

        # 환경 step
        obs, reward, done, truncated, info = env.step(action_text)

        # Tool/User response 토큰 추가 (loss mask = 0)
        if obs:
            obs_tokens = tokenize(obs)
            sample.tokens.extend(obs_tokens)
            sample.loss_mask.extend([0] * len(obs_tokens))
            sample.response += obs

        if truncated:
            sample.status = Sample.Status.TRUNCATED
            break

    sample.reward = reward  # 0.0 or 1.0
    sample.response_length = sum(sample.loss_mask)  # agent response만
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        'task_id': task_id,
        'success': reward == 1.0,
        'num_turns': info.get('num_turns', 0),
    }

    return sample
```

### 6.3 Custom Reward Model

```python
# examples/tau2_bench/reward.py

async def tau2_reward_model(args, sample):
    """Binary reward: 성공=1.0, 실패=0.0"""
    if sample.status == Sample.Status.TRUNCATED:
        return 0.0

    # τ²-Bench는 환경에서 이미 reward 계산됨
    return sample.reward
```

### 6.4 Dynamic Filter (Rejection Sampling)

```python
# examples/tau2_bench/filter.py

def mixed_batch_filter(args, sample_group):
    """전부 성공 또는 전부 실패 배치 제외"""
    rewards = [s.reward for s in sample_group]

    all_success = all(r == 1.0 for r in rewards)
    all_failure = all(r == 0.0 for r in rewards)

    if all_success or all_failure:
        return DynamicFilterOutput(keep=False, reason="no_variance")

    return DynamicFilterOutput(keep=True)
```

### 6.5 학습 명령어

```bash
python train.py \
    # 커스텀 함수
    --custom-generate-function-path "examples.tau2_bench.generate:generate" \
    --custom-rm-path "examples.tau2_bench.reward:tau2_reward_model" \
    --dynamic-sampling-filter-path "examples.tau2_bench.filter:mixed_batch_filter" \
    \
    # 데이터
    --prompt-data /path/to/task_ids.jsonl \
    --input-key task_id \
    \
    # Rollout 설정
    --rollout-batch-size 16 \
    --n-samples-per-prompt 4 \
    --over-sampling-batch-size 32 \
    \
    # 학습 설정
    --global-batch-size 64 \
    --num-steps-per-rollout 1 \
    --num-rollout 1000 \
    \
    # Advantage Estimator (GRPO)
    --advantage-estimator grpo \
    --grpo-std-normalization \
    --normalize-advantages \
    \
    # RL 하이퍼파라미터
    --gamma 1.0 \
    --lambd 1.0 \
    --eps-clip 0.2 \
    --kl-coef 0.0 \
    --entropy-coef 0.0 \
    \
    # 모델 설정
    --load /path/to/base_model \
    --save /path/to/output \
    --save-interval 100
```

---

## 7. 우리 RLVR 설계와의 매핑

| 우리 설계 | Slime 구현 |
|----------|-----------|
| `gamma = 1.0` | `--gamma 1.0` |
| `lambda = 1.0` | `--lambd 1.0` |
| `sample_unit = step` | `loss_mask`로 턴별 마스킹 |
| `advantage = G - mean(G)` | `--grpo-std-normalization` (mean only) 또는 커스텀 |
| `think_format = 기존 유지` | 마지막 턴만 think 토큰 포함 |
| `rejection = mixed batch` | `--dynamic-sampling-filter-path` |
| Binary reward (0/1) | Custom reward model |

### 차이점 및 조정 필요

1. **Mean centering vs STD normalization**
   - 우리 설계: `A = G - mean(G)` (std 안 나눔)
   - Slime GRPO: `--grpo-std-normalization`은 std도 나눔
   - 해결: 커스텀 advantage function 작성 또는 `--grpo-std-normalization` 없이 사용

2. **Step 기반 샘플링**
   - 우리 설계: 각 턴이 독립 샘플
   - Slime: Trajectory 단위가 기본
   - 해결: `loss_mask`로 각 턴의 agent response만 학습

---

## 8. 핵심 설정 옵션

### Rollout 설정

| 옵션 | 설명 | 권장값 |
|-----|------|-------|
| `--rollout-batch-size` | 프롬프트 수 | 16-32 |
| `--n-samples-per-prompt` | 프롬프트당 샘플 수 | 4-8 |
| `--over-sampling-batch-size` | 오버샘플링 크기 | rollout-batch-size × 2 |
| `--rollout-max-response-len` | 최대 응답 길이 | 2048+ |

### Advantage Estimator

| 옵션 | 설명 | 권장값 |
|-----|------|-------|
| `--advantage-estimator` | 방식 선택 | `grpo` |
| `--grpo-std-normalization` | STD 정규화 | 선택적 |
| `--normalize-advantages` | Whiten advantages | True |
| `--gamma` | Discount factor | 1.0 |
| `--lambd` | GAE lambda | 1.0 |

### 학습 설정

| 옵션 | 설명 | 권장값 |
|-----|------|-------|
| `--global-batch-size` | 배치 크기 | 64 |
| `--num-steps-per-rollout` | Rollout당 학습 스텝 | 1 (on-policy) |
| `--eps-clip` | PPO clipping | 0.2 |
| `--kl-coef` | KL penalty | 0.0 (binary reward) |
| `--entropy-coef` | Entropy bonus | 0.0 |

---

## 9. 학습 파이프라인 흐름

```
for rollout_id in range(num_rollout):

    1. WEIGHT SYNC
       └─ Megatron weights → SGLang engines

    2. DATA GENERATION (RolloutManager)
       ├─ 각 태스크에 대해 n_samples_per_prompt 샘플 생성
       ├─ τ²-Bench 환경과 multi-turn 상호작용
       ├─ Dynamic filter로 mixed batch만 유지
       └─ Data Buffer에 저장

    3. TRAINING
       ├─ Batch 구성 (packed sequences)
       ├─ Advantage 계산 (GRPO)
       │  └─ A = reward - group_mean
       ├─ Policy loss 계산
       │  └─ L = -A * log_prob (PPO clipping 적용)
       ├─ Backward & optimizer step
       └─ 메트릭 로깅

    4. CHECKPOINT (optional)
       └─ 모델 저장
```

---

## 10. 참고 자료

- GitHub: https://github.com/THUDM/slime
- τ-bench 예제: `examples/tau-bench/`
- Megatron 학습: `slime/backends/megatron_utils/`
- Advantage 계산: `slime/utils/ppo_utils.py`
