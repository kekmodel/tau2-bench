# τ²-bench 논문 리뷰 및 RLVR 적용 방안

## 1. 논문 개요

### 기본 정보
- **논문명**: τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment
- **arXiv**: 2506.07982v1 (2025년 6월 9일)
- **상태**: Preprint, Under review
- **코드**: https://github.com/sierra-research/tau2-bench

### 핵심 기여
1. **이중 제어 도메인 (Dual-Control)**: 에이전트와 사용자 모두 도구 사용 가능
2. **조합적 태스크 생성**: 원자적 서브태스크 조합 → 2,285개 태스크 자동 생성
3. **신뢰성 있는 사용자 시뮬레이터**: 도구로 행동 제약 → 오류율 47% → 16% 감소
4. **세밀한 오류 분석**: 추론 오류 vs 협업 오류 분리 가능

---

## 2. 기술적 프레임워크: Dec-POMDP

### 정의
**(S, {Aᵢ}, {Oᵢ}, T, R, U, M)**

| 구성요소 | 설명 |
|----------|------|
| S | 상태 = S_world (에이전트 DB + 사용자 DB) ⊗ S_history |
| Aᵢ | 행동 = 도구 호출 또는 메시지 |
| Oᵢ | 관측 = 도구 결과 또는 상대방 메시지 |
| T | 전이 함수 |
| R | 보상 함수 (assert 기반) |
| U | 지시 공간 (정책 문서 포함) |
| M | 메시지 공간 |

### 에이전트의 사용자 도구 인식
- 에이전트는 사용자 도구 종류를 **정책 문서(시스템 프롬프트)**로 알고 있음
- 도구 호출 결과는 모름 (메시지로 전달받아야 함)

---

## 3. 데이터 규모

| 도메인 | 에이전트 도구 | 사용자 도구 | 태스크 수 |
|--------|-------------|------------|----------|
| Retail | 13개 | 없음 | 115개 |
| Airline | 12개 | 없음 | 50개 |
| **Telecom** | 13개 | **30개** | 114개 (전체 2,285개) |

---

## 4. 평가 방법

### 평가 지표
- **pass^k**: k번 독립 실행 중 성공 비율

### Telecom 도메인 평가 방식
- **Status Assertions만 사용**
- 최종 사용자 DB 상태만 체크
- 도구 호출 순서, 중간 과정 체크 안 함

```python
# 예시
assert_service_status(expected_status="connected")  # True/False
```

### 사용자 시뮬레이터 오류 처리
- 사전 방지: 도구로 행동 제약
- 사후: 인간이 검토하여 치명적/양성 오류 분류
- 명시적 점수 보정 없음 (오류율 별도 보고)

---

## 5. 실험 결과 핵심

### 모드별 성능 (Telecom, pass^1)

| 모델 | Default | No-User | 차이 |
|------|---------|---------|------|
| gpt-4.1 | 34% | 52% | -18% |
| o4-mini | 42% | 67% | -25% |

**결론**: 협업/커뮤니케이션이 현재 LLM의 주요 병목

---

## 6. RLVR 적용 방안 (채택된 최종 설계)

### 보상 구조

```python
# 최종 채택
reward = 1 if (success and turns <= MAX_TURNS) else 0
gamma = 1.0  # 디스카운트 없음
```

### 설계 원칙

| 원칙 | 결정 |
|------|------|
| 중간 보상 | ❌ 없음 (Reward hacking 방지) |
| Discount factor | 1.0 (최적 경로 손해 방지) |
| 효율성 강제 | MAX_TURNS 초과 시 실패 처리 |
| 검증 방식 | Assert 함수 (Verifiable) |

### MAX_TURNS 설정 가이드

```python
MAX_TURNS = optimal_turns * 2  # 최적의 2배까지 허용

# 또는 태스크 유형별
MAX_TURNS = {
    "service_issue": 10,
    "mobile_data_issue": 15,
    "mms_issue": 20,
}
```

### 왜 이 설계인가?

| 대안 | 문제점 | 채택 여부 |
|------|--------|----------|
| Dense reward | Reward hacking 쉬움 | ❌ |
| Discount < 1 | 최적 경로도 손해 | ❌ |
| Sparse + gamma=1 | 비효율 행동 구분 안 됨 | ❌ |
| **Sparse + gamma=1 + MAX_TURNS** | 깔끔, 해킹 없음 | ✅ |

### Value Function 역할

```python
# V(s) = "이 상태에서 MAX_TURNS 내에 성공할 확률"
# Advantage A(s,a) = Q(s,a) - V(s)
# → 좋은 행동: A > 0 → 확률 증가
# → 나쁜 행동: A < 0 → 확률 감소
```

Credit assignment를 Value function이 자연스럽게 해결

### GAE Lambda 설정

```python
gamma = 1.0   # 디스카운트 없음
lambda = 1.0  # Monte Carlo return
```

#### λ=1의 의미

GAE 공식에서 γ=1, λ=1이면:
```python
A_GAE = δ_t + δ_{t+1} + δ_{t+2} + ... + δ_T
      = G_t - V(s_t)  # 실제 리턴 - 현재 가치 추정
```

Sparse reward (마지막에만 0 or 1)이므로:
```python
A(s_t, a_t) = (0 or 1) - V(s_t)
```

모든 스텝의 행동이 **최종 성공/실패에 동일하게 기여**한 것으로 취급됨.

#### 왜 λ=1인가?

| λ 값 | 특성 | 적합성 |
|------|------|--------|
| λ=0 | 1-step TD, high bias | ❌ sparse reward에서 신호 전파 안 됨 |
| λ=0.95 | 균형 | △ 일반적 선택이지만 굳이 필요 없음 |
| **λ=1** | Monte Carlo, no bias | ✅ 알파고와 동일한 방식, sparse reward에 적합 |

- Sparse reward → 중간에 신호 없음 → 최종 결과로만 판단
- 알파고도 승패(+1/-1)로만 Value network 학습, 잘 작동함
- Variance가 높을 수 있지만, rollout 수로 보완

### 학습 안정성을 위한 Rejection Sampling

```python
# Rollout 결과 필터링
# 전부 성공 or 전부 실패인 배치는 제외
# → Advantage 분산 확보 → 학습 신호 유지
```

### Curriculum Learning

논문의 평균 턴 수 기준 난이도 활용:
```python
difficulty_by_avg_turns = {
    "easy": tasks with avg_turns < 5,
    "medium": tasks with 5 <= avg_turns < 10,
    "hard": tasks with avg_turns >= 10,
}
# easy → medium → hard 순서로 학습
```

---

## 7. 뱅킹 도메인 설계 계획

### 이중 제어 구조

| 역할 | 에이전트 (은행 상담사) | 사용자 (고객) |
|------|----------------------|--------------|
| DB | 계좌, 대출, 카드, 환율 | 모바일 뱅킹 앱 상태 |
| 도구 | 조회, 승인, 처리 | 앱에서 직접 조작 |

### 5개 서브도메인

| 도메인 | 에이전트 도구 예시 | 사용자 도구 예시 |
|--------|-------------------|-----------------|
| **송금** | verify_recipient, approve_transfer | enter_amount, confirm_otp |
| **저축** | get_savings_products, open_account | select_product, set_auto_transfer |
| **대출** | check_credit_score, approve_loan | upload_documents, sign_contract |
| **외환** | get_exchange_rate, process_exchange | select_currency, confirm_exchange |
| **카드** | check_eligibility, issue_card | select_card, activate_card |

### 조합적 태스크 생성

```python
subtasks = {
    "otp_expired": (set_otp_expired, user_request_new_otp, assert_otp_valid),
    "daily_limit_exceeded": (set_over_limit, agent_increase_limit, assert_transfer_possible),
    "recipient_not_registered": (clear_recipients, user_add_recipient, assert_recipient_exists),
    ...
}

# 조합 → 복합 시나리오
task = otp_expired + daily_limit_exceeded
```

### 평가

```python
reward = 1 if (assert_task_complete(state) and turns <= MAX_TURNS) else 0
```

---

## 8. 핵심 인사이트 요약

1. **이중 제어가 시뮬레이션을 더 쉽게 만든다** (역설적)
   - 사용자 도구 → 행동 제약 → 오류 감소

2. **협업이 추론보다 어렵다**
   - No-User → Default 전환 시 18-25% 성능 하락

3. **RLVR은 단순하게**
   - Binary reward + gamma=1 + MAX_TURNS
   - 타협 없이 깔끔하게

4. **Assert 함수의 다용도 활용**
   - 평가: 최종 상태 검증
   - RL: Verifiable reward
   - 필터: Rejection sampling

---

## 9. 참고 자료

- τ²-bench GitHub: https://github.com/sierra-research/tau2-bench
- 원본 τ-bench 논문: arXiv:2406.12045
- Dec-POMDP: Oliehoek & Amato (2016)
