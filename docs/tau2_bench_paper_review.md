# τ²-Bench 논문 상세 리뷰

## 논문 정보

- **제목**: τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment
- **저자**: Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan
- **소속**: Sierra, University of Toronto, Vector Institute
- **arXiv**: 2506.07982
- **코드**: https://github.com/sierra-research/tau2-bench

---

## 1. 연구 동기

### 1.1 기존 벤치마크의 한계

기존 대화형 AI 에이전트 벤치마크(τ-bench, WebArena, SWE-bench 등)는 **단일 제어(Single-Control)** 환경:

```
사용자 ──(정보 제공)──▶ 에이전트 ──(도구)──▶ 환경
```

- 사용자는 수동적 정보 제공자
- 에이전트만 환경 조작

### 1.2 현실 세계의 이중 제어

기술 지원, 의료 상담 등에서는 **사용자가 직접 행동**해야 함:

```
에이전트 ──(도구)──▶ 에이전트 환경 (DB, 시스템)
    │
    │◀────(대화)────▶│
                     │
사용자 ──(직접 조작)──▶ 사용자 환경 (휴대폰 등)
```

예시:
```
고객: "인터넷이 안 돼요"
에이전트: (계정 조회) "계정은 정상이네요. 비행기 모드 아이콘 보이시나요?"
고객: "네, 보여요"
에이전트: "비행기 모드 꺼주시겠어요?"
고객: (직접 끔) "껐어요"
에이전트: "이제 되시나요?"
```

이 **협력적 문제 해결** 능력을 평가하는 벤치마크가 없었음.

---

## 2. 이론적 프레임워크: Dec-POMDP

### 2.1 정의

**Decentralized Partially Observable Markov Decision Process**

다중 에이전트가 부분적 관찰만으로 협력해야 하는 문제 모델링.

```
Dec-POMDP = (I, S, {Aᵢ}, T, {Ωᵢ}, O, R)
```

| 기호 | 의미 | τ²-Bench 구현 |
|-----|------|--------------|
| I | 에이전트 집합 | {AI 에이전트, 사용자} |
| S | 상태 공간 | 에이전트 DB + 사용자 환경 + 대화 히스토리 |
| Aᵢ | 행동 공간 | 도구 호출 ∪ 자연어 메시지 |
| T | 전이 함수 | 도구 실행 결과로 상태 변경 |
| Ωᵢ | 관찰 공간 | 도구 결과 또는 상대방 메시지 |
| O | 관찰 함수 | 자신의 도구 결과만 관찰 가능 |
| R | 보상 함수 | 작업 완료시 1, 실패시 0 |

### 2.2 핵심 제약

**한 턴에 하나의 행동만 허용**:

```python
action = agent.decide()
if is_tool_call(action):
    result = execute_tool(action)
    # 사용자에게 전달 안 됨
elif is_message(action):
    user.receive(action)
    # 도구 실행 안 됨
```

### 2.3 정보 비대칭

| 에이전트 관찰 가능 | 사용자 관찰 가능 |
|------------------|----------------|
| 고객 계정 정보 | 휴대폰 화면 |
| 요금제 상세 | 설정 메뉴 상태 |
| 서비스 상태 (백엔드) | 오류 메시지 |
| 정책 문서 | 신호 강도 |

공유: 대화 히스토리만

에이전트는 사용자 환경을 **직접 볼 수 없고**, 물어봐야 함.

### 2.4 Partially Decoupled MDP (논문 정의의 한계)

논문에서는 **Decoupled MDP**로 정의하지만, 실제로는 **비대칭적 결합** 구조:

**User (완전 독립):**
- Agent의 tool을 **전혀 모름**
- 그냥 "문제가 있다"고 말하고, Agent가 해결해주길 기대
- Agent가 뭘 할 수 있는지 관심 없음

**Agent (약하게 결합):**
- User의 tool API는 모름 (`toggle_airplane_mode()` 같은 함수명)
- 하지만 User가 **뭘 할 수 있는지** 대략 앎 (domain knowledge via system prompt)
- "사용자가 비행기 모드를 끌 수 있다"는 자연어 수준의 지식 보유

```
Agent knows: A_π (정확히) + description(A_μ) (자연어로 대략)
User knows: A_μ (정확히) + nothing about A_π
```

**더 정확한 표현:**
- **Asymmetric Information MDP** 또는
- **Partially Decoupled MDP**

현실적으로 완전한 분리는 불가능 - 협력하려면 상대가 뭘 할 수 있는지 어느 정도는 알아야 함.
Agent는 User를 **지시/안내**해야 하므로 User capability의 자연어 description이 필요.

---

## 3. 벤치마크 구성

### 3.1 도메인 구조

| 도메인 | 제어 유형 | 에이전트 도구 | 사용자 도구 | 태스크 수 |
|--------|----------|-------------|-----------|----------|
| Retail | 단일 | 13개 | 없음 | 115개 |
| Airline | 단일 | 12개 | 없음 | 50개 |
| **Telecom** | **이중** | 13개 | **30개** | 114개 (2,285개 가능) |

### 3.2 작업 생성 파이프라인 (5단계)

**1단계**: LLM으로 PRD(제품 요구사항 문서) 생성

**2단계**: 사용자 환경 정의 (가상 스마트폰)

```python
class VirtualPhone:
    def __init__(self):
        self.airplane_mode = False
        self.wifi_enabled = True
        self.mobile_data_enabled = True
        self.signal_strength = 4

    def toggle_airplane_mode(self):
        self.airplane_mode = not self.airplane_mode
        if self.airplane_mode:
            self.wifi_enabled = False
            self.mobile_data_enabled = False
```

**3단계**: 원자적 작업 정의 (3함수 구조)

```python
class AirplaneModeIssue:
    @staticmethod
    def f_init(phone, db):
        """문제 상태 초기화"""
        phone.airplane_mode = True

    @staticmethod
    def f_solve(phone, db):
        """해결 절차"""
        phone.toggle_airplane_mode()

    @staticmethod
    def f_assert(phone, db) -> bool:
        """성공 검증"""
        return phone.airplane_mode == False
```

**4단계**: 원자 작업 조합 → 복합 작업 생성 (2,285개)

**5단계**: 정책 문서 생성 및 수동 검수

### 3.3 사용자 의도 유형

| 의도 | 복잡도 | 설명 |
|-----|-------|------|
| service_issue | 낮음 | 단순 서비스 문의 |
| mobile_data_issue | 중간 | 데이터 연결 문제 |
| mms_issue | 높음 | MMS 전송 문제 (다단계 진단) |

---

## 4. 사용자 시뮬레이터

### 4.1 기존 방식의 문제

자연어 프롬프트로만 제어:
- LLM이 시나리오를 "잊어버림" (hallucination)
- 불가능한 행동 수행
- 일관성 없는 응답

**오류율: 40-47%** (치명적 오류 12-13%)

### 4.2 τ²-Bench 해결책: 도구 기반 제약

```python
class ConstrainedUserSimulator:
    def __init__(self, phone, scenario):
        self.phone = phone

    def get_available_tools(self):
        """현재 상태에서 가능한 행동만 반환"""
        tools = [self.toggle_airplane_mode, self.restart_phone]
        if not self.phone.airplane_mode:
            tools.append(self.toggle_wifi)
        return tools
```

핵심 개선:
1. 정의된 도구만 사용 가능
2. 실제 환경 상태 반영
3. 불가능한 행동 시스템 레벨 차단

**오류율: 16%** (치명적 오류 6%) - **60% 감소**

---

## 5. 평가

### 5.1 pass^k 메트릭

```python
def pass_at_k(results, k):
    """k번 중 적어도 1번 성공 확률"""
    n = len(results)
    c = sum(results)
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))
```

### 5.2 검증 방식

Telecom 도메인: **Status Assertions만 사용**
- 최종 사용자 환경 상태 체크
- 도구 호출 순서, 중간 과정 무관

```python
assert phone.airplane_mode == False
assert phone.can_use_mobile_data() == True
```

---

## 6. 실험 결과

### 6.1 모델별 성능

| 모델 | Retail | Airline | Telecom |
|-----|--------|---------|---------|
| GPT-4.1 | 74% | 56% | **34%** |
| GPT-4.1-mini | 46% | 37% | 30% |
| o4-mini | 50% | 50% | 42% |
| Claude-3.7-Sonnet | 60% | 50% | **49%** |

**Telecom(이중 제어)이 가장 어려움**

### 6.2 핵심 Ablation: 추론 vs 조정

| 모드 | 설명 | GPT-4.1 성능 |
|-----|------|-------------|
| No-User | 에이전트가 사용자 환경도 직접 조작 | 52% |
| Default | 사용자에게 지시 | 34% |
| Oracle Plan | 정확한 계획 제공 | 59% |

**핵심 발견**:
```
No-User(52%) → Default(34%): -18%
```
이 갭이 순수하게 **사용자 조정 능력 부족** 때문

### 6.3 복잡도별 성능

| 필요 행동 수 | No-User | Default |
|------------|---------|---------|
| 1-3 | 75% | 55% |
| 4-6 | 45% | 25% |
| 7+ | 10% | 5% |

**7개 이상 행동 필요시 거의 실패**

### 6.4 페르소나 영향

| 페르소나 | 성능 |
|---------|-----|
| Easy (41세, 기술 친숙) | 40% |
| Hard (64세, 기술 약함) | 30% |
| None | 32% |

---

## 7. 핵심 통찰

### 7.1 이중 제어가 시뮬레이션을 쉽게 만든다 (역설)

- 사용자 도구 → 행동 제약 → 오류 감소
- 자연어만으로 제어하는 것보다 도구 기반이 안정적

### 7.2 협업이 추론보다 어렵다

- No-User → Default: 18-25% 성능 하락
- 현재 LLM의 주요 병목: 사용자 지도 및 조정 능력

### 7.3 복잡도의 한계

- 7개 이상 행동 필요시 급격한 성능 저하
- 장기 태스크 처리 능력 부족

### 7.4 Assert 함수의 다용도 활용

- 평가: 최종 상태 검증
- RL: Verifiable reward
- 필터: Rejection sampling

---

## 8. 한계점

1. **도메인 한정**: 이중 제어는 Telecom 1개뿐
2. **전문가-초보자 갭**: 멘탈 모델 차이 명시적 모델링 없음
3. **확장 비용**: 새 도메인 추가는 노동 집약적

---

## 9. RLVR 적용 관점에서의 의의

1. **Verifiable Reward**: Assert 함수로 객관적 성공/실패 판정 가능
2. **구성적 태스크 생성**: 난이도 조절 가능 → Curriculum Learning
3. **신뢰할 수 있는 시뮬레이터**: 도구 기반 제약 → 학습 안정성
4. **턴 수 기반 난이도**: 이미 정의됨 → 그대로 활용 가능
