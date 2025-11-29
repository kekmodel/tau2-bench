# τ²-Bench 프로젝트 분석

## 분석 일자
2025-11-29

---

## 1. 프로젝트 개요

τ²-Bench는 **이중 제어(Dual-Control) 환경**에서 대화형 AI 에이전트를 평가하기 위한 벤치마크입니다.

### 핵심 특징
- **Dec-POMDP 프레임워크**: 에이전트와 사용자가 각자의 환경을 제어
- **협력적 문제 해결**: 에이전트가 사용자를 가이드하여 문제 해결
- **Verifiable Reward**: Assert 함수로 객관적 성공/실패 판정

### 디렉토리 구조

```
tau2-bench/
├── src/tau2/
│   ├── domains/           # 도메인별 구현
│   │   ├── airline/       # 단일 제어
│   │   ├── retail/        # 단일 제어
│   │   └── telecom/       # 이중 제어 (핵심)
│   ├── orchestrator/      # Dec-POMDP 메시지 라우팅
│   ├── user/              # 사용자 시뮬레이터
│   ├── evaluator/         # Assert 기반 평가
│   ├── gym/               # Gymnasium 인터페이스
│   └── data_model/        # 데이터 모델
├── data/tau2/             # 도메인 데이터, 정책 문서
└── docs/                  # 문서
```

---

## 2. 도메인 구조

| 도메인 | 제어 유형 | 에이전트 도구 | 사용자 도구 | 태스크 수 |
|--------|----------|-------------|-----------|----------|
| Retail | 단일 | 13개 | 없음 | 115개 |
| Airline | 단일 | 12개 | 없음 | 50개 |
| **Telecom** | **이중** | 13개 | **30개** | 114개 (2,285개 가능) |

### Telecom 도메인 (이중 제어 핵심)

```
에이전트 ──(도구)──▶ 백엔드 시스템 (DB, 계정 관리)
    │
    │◀────(대화)────▶│
                     │
사용자 ──(도구)──▶ 가상 스마트폰 (설정, 네트워크)
```

---

## 3. 핵심 컴포넌트 분석

### 3.1 태스크 생성 (3-함수 구조)

**위치**: `src/tau2/domains/telecom/tasks/`

```python
class MobileDataTask(BaseTask):
    def init_funcs(self) -> list[EnvFunction]:
        """문제 상태 초기화"""
        return [lambda env: env.user_env.toggle_airplane_mode()]

    def fix_funcs(self) -> list[EnvFunction]:
        """해결 절차 (정답)"""
        return [lambda env: env.user_env.toggle_airplane_mode()]

    def env_assertions(self) -> list[EnvAssertion]:
        """성공 검증"""
        return [lambda env: env.user_env.airplane_mode == False]
```

### 3.2 사용자 시뮬레이터

**위치**: `src/tau2/user/user_simulator.py`

```python
class UserSimulator(BaseUser):
    def __init__(self, tools, instructions, llm, llm_args):
        self.tools = tools  # 사용자 도구 (30개)

    def generate_next_message(self, message, state):
        # LLM이 도구 기반으로 응답 생성
        # 불가능한 행동은 시스템 레벨에서 차단
```

**도구 기반 제약의 효과**:
- 자연어만 사용: 오류율 40-47%
- 도구 기반: 오류율 16% (60% 감소)

### 3.3 Orchestrator (Dec-POMDP 구현)

**위치**: `src/tau2/orchestrator/orchestrator.py`

메시지 라우팅 로직:
```python
# 에이전트 → 사용자 (텍스트 메시지)
if message.is_text() and from_role == AGENT:
    route_to(USER)

# 에이전트 → 환경 (도구 호출)
if message.is_tool_call() and from_role == AGENT:
    route_to(ENV)

# 사용자 → 사용자 환경 (사용자 도구)
if message.is_tool_call() and from_role == USER:
    route_to(USER_ENV)
```

### 3.4 평가자 (Evaluator)

**위치**: `src/tau2/evaluator/evaluator_env.py`

```python
def evaluate(self, env_state) -> float:
    """Assert 함수로 성공 여부 판정"""
    for assertion in task.env_assertions():
        if not assertion(env_state):
            return 0.0  # 실패
    return 1.0  # 성공
```

### 3.5 Gymnasium 인터페이스

**위치**: `src/tau2/gym/gym_agent.py`

```python
class AgentGymEnv(gym.Env):
    """에이전트 관점의 Gym 환경"""

    def reset(self):
        # 에피소드 시작, 초기 관찰 반환

    def step(self, action: str):
        # action: 텍스트 메시지 또는 도구 호출 문자열
        # returns: (obs, reward, done, truncated, info)
```

---

## 4. 사용자 도구 (30개)

**위치**: `src/tau2/domains/telecom/user_tools.py`

### 카테고리별 분류

| 카테고리 | 도구 예시 |
|---------|----------|
| 상태 확인 | `check_status_bar`, `check_signal_strength` |
| 네트워크 | `toggle_airplane_mode`, `toggle_wifi`, `toggle_mobile_data` |
| 설정 | `open_settings`, `navigate_to_*` |
| APN | `check_apn_settings`, `reset_apn_to_default` |
| 시스템 | `restart_phone`, `check_software_update` |

### 도구 상태 의존성

```python
def get_available_tools(phone_state):
    tools = [check_status_bar, restart_phone]

    if not phone_state.airplane_mode:
        tools.append(toggle_wifi)
        tools.append(toggle_mobile_data)

    return tools
```

---

## 5. 에이전트 도구 (13개)

**위치**: `src/tau2/domains/telecom/tools.py`

| 도구 | 설명 |
|-----|------|
| `get_customer_by_phone` | 전화번호로 고객 조회 |
| `get_line_details` | 회선 상세 정보 |
| `get_account_balance` | 계정 잔액 확인 |
| `check_network_status` | 네트워크 상태 확인 |
| `create_support_ticket` | 지원 티켓 생성 |
| ... | |

---

## 6. 태스크 유형 및 난이도

### 의도(Intent) 유형

| 의도 | 복잡도 | 필요 행동 수 |
|-----|-------|------------|
| `service_issue` | 낮음 | 1-3 |
| `mobile_data_issue` | 중간 | 3-6 |
| `mms_issue` | 높음 | 5-10+ |

### 태스크 ID 형식

```
[intent]problem1|problem2[PERSONA:name]

예시:
[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]
[mms_issue]airplane_mode_on|mms_disabled[PERSONA:alex]
```

### 복합 태스크 생성

```python
class ComposedTask:
    """여러 원자 태스크 조합"""
    def __init__(self, atomic_tasks: list[BaseTask]):
        self.tasks = atomic_tasks

    def init_funcs(self):
        return [f for t in self.tasks for f in t.init_funcs()]

    def env_assertions(self):
        return [a for t in self.tasks for a in t.env_assertions()]
```

---

## 7. Gymnasium 환경 사용법

### 기본 사용

```python
from tau2.gym.gym_agent import AgentGymEnv, register_gym_agent

register_gym_agent()

env = AgentGymEnv(
    domain='telecom',
    task_id='[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]',
    max_steps=30,
    solo_mode=False,  # True=에이전트만, False=사용자 시뮬레이터 포함
    user_llm='claude-haiku-4-5-20251001',
    user_llm_args={'temperature': 0.0},
)

obs, info = env.reset()

# 액션 타입 1: 텍스트 메시지 (사용자에게 전달)
obs, reward, done, truncated, info = env.step("Hello! How can I help?")

# 액션 타입 2: 도구 호출 (환경 실행)
obs, reward, done, truncated, info = env.step("get_customer_by_phone(phone_number='555-123-2002')")
```

### 파라미터 설명

| 파라미터 | 설명 |
|---------|------|
| `domain` | 'telecom', 'retail', 'airline' |
| `task_id` | 태스크 식별자 |
| `max_steps` | 최대 스텝 수 |
| `solo_mode` | True면 사용자 시뮬레이터 없이 에이전트만 |
| `user_llm` | 사용자 시뮬레이터 모델 |
| `user_llm_args` | LLM 추가 인자 |

### 반환값

| 값 | 설명 |
|---|------|
| `obs` | 마지막 메시지 (사용자 응답 또는 도구 결과) |
| `reward` | 0.0 (진행 중) 또는 1.0 (성공) |
| `done` | 에피소드 종료 여부 |
| `truncated` | max_steps 도달 여부 |
| `info` | 추가 정보 (히스토리 등) |

---

## 8. 평가 실행

### CLI 사용

```bash
# 전체 도메인 평가
.venv/bin/python -m tau2.run_eval \
    --domain telecom \
    --agent llm_agent \
    --agent_llm claude-sonnet-4-20250514 \
    --user_llm claude-haiku-4-5-20251001 \
    --task_set telecom_small

# 특정 태스크만
.venv/bin/python -m tau2.run_eval \
    --domain telecom \
    --task_id '[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]'
```

### 태스크 조회

```python
from tau2.registry import get_task_set

tasks = get_task_set('telecom_small')
for task in tasks[:5]:
    print(task.task_id)
```

---

## 9. 실험 결과 (논문 기준)

### 모델별 성능

| 모델 | Retail | Airline | Telecom |
|-----|--------|---------|---------|
| GPT-4.1 | 74% | 56% | **34%** |
| Claude-3.7-Sonnet | 60% | 50% | **49%** |
| o4-mini | 50% | 50% | 42% |

### 핵심 발견

1. **이중 제어가 가장 어려움**: Telecom 성능이 일관되게 낮음
2. **협업 > 추론**: No-User(52%) → Default(34%) = -18% 갭
3. **복잡도 한계**: 7개+ 행동 필요시 급격한 성능 저하

---

## 10. RLVR 적용 관점

### 왜 τ²-Bench가 RLVR에 적합한가?

1. **Verifiable Reward**: Assert 함수로 객관적 0/1 판정
2. **구성적 태스크**: 난이도 조절 가능 → Curriculum Learning
3. **신뢰할 수 있는 시뮬레이터**: 도구 기반 → 학습 안정성
4. **Multi-turn 구조**: Credit assignment 연구에 적합

### 설계 결정 (별도 문서 참조)

- `docs/rlvr_final_design.md`: 최종 RLVR 설계
- `docs/rlvr_design_discussion.md`: 설계 논의 과정

---

## 11. 주요 파일 경로

| 파일 | 설명 |
|-----|------|
| `src/tau2/gym/gym_agent.py` | Gymnasium 환경 |
| `src/tau2/orchestrator/orchestrator.py` | Dec-POMDP 메시지 라우팅 |
| `src/tau2/user/user_simulator.py` | 사용자 시뮬레이터 |
| `src/tau2/evaluator/evaluator_env.py` | Assert 기반 평가 |
| `src/tau2/domains/telecom/tasks/` | Telecom 태스크 정의 |
| `src/tau2/domains/telecom/user_tools.py` | 사용자 도구 (30개) |
| `src/tau2/domains/telecom/tools.py` | 에이전트 도구 (13개) |
| `data/tau2/telecom/` | 정책 문서, 데이터 |
