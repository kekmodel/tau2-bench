# Slime + Tau2 Step 기반 학습 구현 계획

## 작성일: 2025-11-30

## 1. 목표

`docs/rlvr_final_design.md`의 설계대로 tau2-bench 환경에서 **Step 단위 샘플링** 기반 RLVR 학습 구현

---

## 2. 설계 요구사항 vs Slime 현황

| 항목 | 설계 요구사항 | Slime 현황 | 작업 |
|------|-------------|-----------|------|
| Sample 단위 | Step (턴별) | Trajectory (전체) | **구현 필요** |
| Think 포맷 | 마지막 턴만 think | 미구현 (매 턴 think 포함) | **구현 필요** |
| Advantage | Mean centering (std 안 나눔) | `--grpo-std-normalization false` | 설정 |
| gamma, lambda | 1.0, 1.0 | `--gamma 1.0 --lambd 1.0` | 설정 |
| Value function | 미사용 | `--use-critic false` | 설정 |
| Rejection | Mixed batch 필터링 | `check_reward_nonzero_std` | 설정 |
| 환경 | tau2 | tau1 (tau_bench) | **구현 필요** |

---

## 3. 파일 구조

```
slime/examples/
├── tau-bench/                    # 기존 (tau1, 참고용)
│   ├── generate_with_tau.py
│   ├── trainable_agents.py
│   ├── openai_tool_adapter.py
│   └── ...
│
└── tau2-bench/                   # 신규 생성
    ├── __init__.py               # [신규]
    ├── generate_with_tau2.py     # [신규] 메인 엔트리포인트
    ├── trainable_agents.py       # [신규] tau2용 학습 가능 에이전트
    ├── step_sampler.py           # [신규] trajectory → step samples 변환
    └── run_train.sh              # [신규] 학습 스크립트 예시
```

---

## 4. 상세 구현 계획

### 4.1 신규 파일: `slime/examples/tau2-bench/__init__.py`

빈 파일 (패키지 인식용)

---

### 4.2 신규 파일: `slime/examples/tau2-bench/step_sampler.py`

**역할**: Trajectory를 Step 단위 Sample 리스트로 변환

```python
# 주요 데이터 구조
@dataclass
class TurnData:
    turn_idx: int
    state_token_ids: List[int]    # 해당 턴 시점까지의 누적 토큰
    action_token_ids: List[int]   # assistant 응답 토큰
    action_loss_mask: List[int]   # action 부분만 1

@dataclass
class StepInteractionResult:
    """기존 InteractionResult 확장"""
    prompt: str
    reward: float
    status: Status
    info: Dict[str, Any]
    turns: List[TurnData]         # 턴별 데이터 (핵심 추가)

# 주요 함수
def trajectory_to_step_samples(
    result: StepInteractionResult,
    task_index: int
) -> List[Sample]:
    """
    1개 trajectory → N개 Sample (턴 수만큼)

    설계 문서:
    - 각 턴의 (state, action)을 독립 Sample로 생성
    - 최종 reward를 모든 턴에 동일하게 적용 (gamma=1.0)
    """
```

**참고**: `slime/examples/tau-bench/trainable_agents.py:26-35` InteractionResult 구조

---

### 4.3 신규 파일: `slime/examples/tau2-bench/trainable_agents.py`

**역할**: tau2 환경용 학습 가능 에이전트

**기반**:
- `slime/examples/tau-bench/trainable_agents.py` (구조 참고)
- `src/tau2/agent/llm_agent.py` (tau2 에이전트 기반)

**주요 수정 사항**:

1. **Import 변경**
   ```python
   # 기존 (tau1)
   from tau_bench.agents.tool_calling_agent import ToolCallingAgent
   from tau_bench.types import Action, RunConfig

   # 변경 (tau2)
   from tau2.agent.llm_agent import LLMAgent
   from tau2.data_model import Action, Message
   ```

2. **`asolve()` 메서드 수정** (Line 176-332 참고)

   **현재 (trajectory 단위)**:
   ```python
   loss_masks = []
   response_token_ids = []
   # 전체 누적 후 하나의 InteractionResult 반환
   # reward는 최종 1개만 저장
   ```

   **변경 (step 단위 + reward 보류/분배)**:
   ```python
   turns_data = []
   cumulative_tokens = prompt_token_ids.copy()

   for step in range(max_num_steps):
       current_state = cumulative_tokens.copy()

       # LLM 호출
       output = await self._call_llm(url, payload)
       response_text = output["text"]
       reasoning = output.get("reasoning_content", "")

       # 마지막 턴 여부 (아직 모름, 환경 응답 후 확인)
       # ... action 실행 ...
       env_response = await self._execute_tool(env, action)
       is_last_turn = env_response.done or (step == max_num_steps - 1)

       # 학습용 토큰 결정
       # 중요: 모든 턴의 학습 샘플에 해당 턴의 think 포함
       # (롤아웃 히스토리는 think 제외, 학습은 각 step을 독립 trajectory로 취급)
       if reasoning:
           action_for_tokens = reasoning + response_text
       else:
           action_for_tokens = response_text

       # 턴별 데이터 저장 (reward는 보류)
       turns_data.append(TurnData(
           turn_idx=step,
           state_token_ids=current_state,
           action_token_ids=tokenize(action_for_tokens),
           action_loss_mask=[1] * len(tokenize(action_for_tokens)),
           reward=None,  # 보류
       ))

       # 히스토리 업데이트 (think 제외)
       messages.append({"role": "assistant", "content": response_text})
       cumulative_tokens.extend(tokenize(response_text) + env_token_ids)

       if env_response.done:
           break

   # Trajectory 종료 후: 최종 reward를 모든 턴에 분배 (gamma=1.0)
   final_reward = env_response.reward
   for turn in turns_data:
       turn.reward = final_reward  # 동일 reward 적용

   # StepInteractionResult에 turns 포함하여 반환
   result.turns = turns_data
   result.reward = final_reward
   ```

3. **tau2 환경 인터페이스 적응**
   - `env.step(action)` → tau2의 환경 API에 맞게 수정
   - Tool 정보 포맷 변환 (tau2 tools → OpenAI format)

4. **Think 태그 처리 (롤아웃 vs 학습)**

   **sglang 서버의 `--reasoning-parser` 옵션**:

   sglang 서버 실행 시 `--reasoning-parser qwen3` 옵션을 사용하면 응답이 자동 분리됨:
   ```python
   # --reasoning-parser 사용 시 sglang 응답 포맷
   output = {
       "text": "최종 응답 (think 제외)",
       "reasoning_content": "<think>...</think> 내용",  # 별도 필드
       "meta_info": {...}
   }
   ```

   참고: [SGLang Structured Outputs for Reasoning Models](https://docs.sglang.ai/advanced_features/structured_outputs_for_reasoning_models.html)

   | sglang 서버 설정 | output["text"] | reasoning_content |
   |-----------------|----------------|-------------------|
   | `--reasoning-parser` 없음 | think 포함 전체 | 없음 |
   | `--reasoning-parser qwen3` | think 제외 응답만 | think 내용 분리 |

   **핵심 설계**:

   **롤아웃 시 (히스토리 누적)**: think 제외
   ```
   턴1: state0 → think+A 생성 → 히스토리에 A만 저장
   턴2: state1 → think+B 생성 → 히스토리에 B만 저장
   턴3: state2 → think+C 생성 → terminal
   ```

   **학습 샘플 (각 step이 독립 trajectory)**: think 포함
   ```
   Sample(state0, think+A, reward)  # 모든 턴에 think 포함!
   Sample(state1, think+B, reward)
   Sample(state2, think+C, reward)
   ```

   **구현 옵션**:

   **옵션 A: `--reasoning-parser` + reasoning_content 활용** (권장)

   sglang 서버 실행:
   ```bash
   python -m sglang.launch_server --model-path <model> --reasoning-parser qwen3
   ```

   클라이언트 코드:
   ```python
   async def asolve(self, ...):
       for step in range(max_num_steps):
           output = await self._call_llm(url, payload)

           # text: think 제외, reasoning_content: think 내용
           response_text = output["text"]           # 히스토리용 (think 없음)
           reasoning = output.get("reasoning_content", "")  # think 내용

           # 히스토리에는 think 없이 추가
           messages.append({"role": "assistant", "content": response_text})

           # 학습용 토큰: 마지막 턴만 think 포함
           if is_last_turn and reasoning:
               action_for_tokens = reasoning + response_text  # think + action
           else:
               action_for_tokens = response_text              # action만
   ```

   **옵션 B: enable_thinking 파라미터 활용** (Qwen3 모델, 서버 설정 없이)
   ```python
   async def asolve(self, ...):
       for step in range(max_num_steps):
           # 마지막 턴 여부에 따라 thinking 제어
           text_input = state.tokenizer.apply_chat_template(
               messages, tokenize=False, add_generation_prompt=True,
               tools=self.tools_info,
               enable_thinking=(step == max_num_steps - 1)  # 마지막만 True
           )
           # ...
   ```

   **옵션 C: Soft Switch 활용** (`/think`, `/no_think`)
   ```python
   # 이전 턴: /no_think, 마지막 턴: /think
   if is_last_turn:
       messages[-1]["content"] += "\n/think"
   else:
       messages[-1]["content"] += "\n/no_think"
   ```

   **권장**: 옵션 A. 서버 레벨에서 깔끔하게 분리됨.

   **주의사항**:
   - `--reasoning-parser`는 OpenAI API 호환이 아님
   - Qwen3-Thinking 모델은 항상 reasoning을 생성 (`enable_thinking=False` 미지원)
   - `</think>` 토큰 ID: 151668

---

### 4.4 신규 파일: `slime/examples/tau2-bench/generate_with_tau2.py`

**역할**: Slime rollout 함수 엔트리포인트

**기반**: `slime/examples/tau-bench/generate_with_tau.py`

**주요 수정 사항**:

1. **Import 변경**
   ```python
   # 기존
   from tau_bench.envs import get_env
   from tau_bench.types import RunConfig

   # 변경
   from tau2.environment import Environment
   from tau2.orchestrator import Orchestrator
   ```

2. **환경 초기화 수정** (Line 127-134 참고)
   ```python
   # tau2 환경 초기화 방식으로 변경
   env = Environment(domain="telecom", task_index=task_index)
   # 또는 Orchestrator 사용
   ```

3. **반환 타입 변경** (Line 100-153 참고)
   ```python
   # 기존
   async def generate(...) -> Sample:
       return res_to_sample(interaction_result, task_index)

   # 변경
   async def generate(...) -> List[Sample]:
       return trajectory_to_step_samples(interaction_result, task_index)
   ```

   **Note**: Slime이 `List[Sample]` 반환을 자동 flatten 처리함 (`rollout.py:158-159`)

---

### 4.5 신규 파일: `slime/examples/tau2-bench/run_train.sh`

**역할**: 학습 실행 스크립트 예시

```bash
python train.py \
    --hf-checkpoint <model_path> \
    --rollout-function-path examples.tau2-bench.generate_with_tau2:generate \
    --eval-function-path examples.tau2-bench.generate_with_tau2:generate_eval \
    \
    # Step 기반 학습 설정 (설계 문서 기준)
    --advantage-estimator grpo \
    --grpo-std-normalization false \      # Mean centering (std 안 나눔)
    --gamma 1.0 \
    --lambd 1.0 \
    --use-critic false \
    \
    # Rejection sampling
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters:check_reward_nonzero_std \
    \
    # 기타 설정
    --rollout-batch-size 32 \
    --num-rollout 100 \
    ...
```

---

## 5. 수정 불필요한 파일 (설정으로 해결)

| 파일 | 이유 |
|------|------|
| `slime/backends/megatron_utils/loss.py` | `--grpo-std-normalization false`로 mean centering 적용 |
| `slime/ray/rollout.py` | `List[Sample]` 반환 이미 지원 (Line 158-159) |
| `slime/rollout/filter_hub/` | `check_reward_nonzero_std` 이미 구현됨 |

---

## 6. 구현 순서

```
1. step_sampler.py 작성
   └─ TurnData, StepInteractionResult 정의
   └─ trajectory_to_step_samples() 구현

2. trainable_agents.py 작성
   └─ tau2 import 적용
   └─ asolve() 턴별 데이터 수집 로직 추가
   └─ tau2 환경 인터페이스 적응

3. generate_with_tau2.py 작성
   └─ tau2 환경 초기화
   └─ List[Sample] 반환하도록 수정

4. run_train.sh 작성
   └─ 설계 문서 기준 설정 적용

5. 테스트
   └─ 단일 trajectory → step samples 변환 테스트
   └─ 전체 학습 파이프라인 테스트
```

---

## 7. 핵심 참고 파일

| 파일 | 참고 내용 |
|------|----------|
| `slime/examples/tau-bench/trainable_agents.py` | `asolve()`, `_get_token_delta()` 구조 |
| `slime/examples/tau-bench/generate_with_tau.py` | `generate()`, `res_to_sample()` 구조 |
| `slime/slime/utils/types.py` | `Sample` 데이터 구조 |
| `slime/slime/ray/rollout.py:158-159` | `List[Sample]` flatten 로직 |
| `src/tau2/agent/llm_agent.py` | tau2 에이전트 인터페이스 |
| `src/tau2/environment/environment.py` | tau2 환경 인터페이스 |

---

## 8. 검증 포인트

- [ ] 턴별 `state_token_ids`가 올바르게 누적되는지
- [ ] `loss_mask`가 assistant 응답 부분만 1인지
- [ ] 동일 trajectory의 모든 턴이 같은 reward를 갖는지
- [ ] `check_reward_nonzero_std`로 mixed batch만 통과하는지
- [ ] advantage가 배치 내 mean centering으로 계산되는지
- [ ] 롤아웃 히스토리(state 누적)에 think 태그가 포함되지 않는지
- [ ] 모든 턴의 학습 샘플 action에 해당 턴의 think가 포함되는지
