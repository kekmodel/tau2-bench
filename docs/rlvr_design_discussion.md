# RLVR 설계 논의 정리

## 논의 일자
2025-11-29

---

## 1. 최종 설계 결정

```python
gamma = 1.0   # 디스카운트 없음
lambda = 1.0  # Monte Carlo return
reward = 1 if (success and turns <= MAX_TURNS) else 0
```

---

## 2. 논의 내용

### 2.1 MAX_TURNS 설정

**결론**: No-User 모드나 사용자 난이도 변수는 고려하지 않음

- No-User vs Default 모드 차이는 실험적 분석용일 뿐
- 실제 학습에서는 Default 모드만 사용
- MAX_TURNS는 태스크 유형별로 고정

### 2.2 Curriculum Learning

**결론**: 논문의 평균 턴 수 기준 난이도 활용

```python
difficulty_by_avg_turns = {
    "easy": avg_turns < 5,
    "medium": 5 <= avg_turns < 10,
    "hard": avg_turns >= 10,
}
```

저자가 이미 턴 수 기반 난이도를 정의해둠 → 그대로 활용

### 2.3 Rejection Sampling

**결론**: 롤아웃에서 전부 성공 or 전부 실패 샘플은 제외

- Advantage 분산 확보
- 학습 신호 유지

### 2.4 사용자 시뮬레이터

**결론**: 논문 방식 그대로 따름 (고정된 도구 기반 시뮬레이터)

- 검증 가능한 방법이 안전함
- Self-play나 복잡한 변형 불필요

### 2.5 Value Function과 남은 턴 수

**초기 제안**: 상태에 remaining_turns를 명시적으로 포함?

**결론**: 불필요

근거:
- 알파고도 최종 승패(+1/-1)로만 Value network 학습
- 디스카운트 없이도 V(s)가 "이 상태에서 이길 확률"을 잘 학습함
- 대화 히스토리 자체가 진행 상황을 암묵적으로 담고 있음
- 오버엔지니어링 → 단순하게 가는 게 맞음

### 2.6 GAE Lambda 설정

**논의 과정**:

Q: Advantage 계산에서 λ를 어떻게 설정?

A: λ=1 (Monte Carlo return)

**λ=1의 의미**:

GAE 공식에서 γ=1, λ=1이면:
```python
A_GAE = δ_t + δ_{t+1} + ... + δ_T
      = G_t - V(s_t)  # 실제 리턴 - 현재 가치 추정
```

Sparse reward (마지막에만 0 or 1)이므로:
```python
A(s_t, a_t) = (0 or 1) - V(s_t)
```

모든 스텝의 행동이 최종 성공/실패에 동일하게 기여한 것으로 취급.

**왜 λ=1인가?**

| λ 값 | 특성 | 적합성 |
|------|------|--------|
| λ=0 | 1-step TD, high bias | ❌ sparse reward에서 신호 전파 안 됨 |
| λ=0.95 | 균형 | △ 일반적이지만 굳이 필요 없음 |
| **λ=1** | Monte Carlo, no bias | ✅ sparse reward에 적합 |

**알파고 사례**:
- 알파고도 승패(+1/-1)로만 Value network 학습
- 디스카운트 없이 최종 결과만으로 잘 학습됨
- τ²-bench도 동일한 구조 → 같은 방식 적용

---

## 3. 최종 설계 요약

| 항목 | 결정 | 근거 |
|------|------|------|
| gamma | 1.0 | 최적 경로 손해 방지 |
| lambda | 1.0 | Sparse reward, Monte Carlo |
| reward | 0 or 1 | Verifiable, reward hacking 방지 |
| MAX_TURNS | 태스크별 고정 | 비효율 행동 제한 |
| 시뮬레이터 | 논문 방식 (도구 기반) | 검증 가능, 안전 |
| Curriculum | 평균 턴 수 기준 | 저자 정의 활용 |
| Rejection | 전부 성공/실패 제외 | 학습 신호 확보 |
| Value state | 남은 턴 미포함 | 알파고 사례, 단순화 |

---

## 4. 참고

- 알파고/알파제로: 최종 승패로 Value network 학습, 잘 작동
- τ²-bench: Sparse reward + Assert 기반 검증
- 두 구조가 유사 → 같은 설계 원칙 적용 가능
