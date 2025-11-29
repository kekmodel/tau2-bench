# RLVR 최종 설계 결정

## 논의 일자
2025-11-29

---

## 1. 최종 설계

```python
gamma = 1.0              # 디스카운트 없음
lambda = 1.0             # Monte Carlo return
sample_unit = step       # 턴 단위 샘플
advantage = G - mean(G)  # mean centering (std 안 나눔)
think_format = 기존 유지  # 마지막 턴만 think
rejection = mixed batch  # 전부 성공/실패 배치 제외
```

---

## 2. 핵심 결정 및 근거

### 2.1 Sample 단위: Step 기반

**Trajectory 단위 vs Step 단위 비교**

```python
# Trajectory 단위
trajectory1: A → B → C → reward=1  (전체가 하나의 샘플)
trajectory2: A → B → X → Y → Z → reward=0

# Step 단위
(state0, action1) → G
(state1, action2) → G
(state2, action3) → G
...
```

**Step 기반 선택 이유**:

1. **이전 action들도 학습됨**
   - Trajectory 기반은 마지막 output만 gradient 받음
   - Step 기반은 각 턴의 action이 독립 샘플로 학습

2. **길이에 따른 implicit bias가 τ²-bench에 적합**

| 상황 | Step 기반 효과 | 합리적? |
|------|---------------|--------|
| 짧게 성공 | 강하게 강화 | ✓ 효율적으로 해결 |
| 길게 성공 | 약하게 강화 | ✓ 비효율적이었음 |
| 짧게 실패 (조기포기) | 강하게 약화 | ✓ 빨리 잘못된 판단 |
| 길게 실패 (MAX_TURNS) | 약하게 약화 | ✓ 노력은 했음 |

### 2.2 Think 포맷: 기존 유지 (마지막만)

**배경**: 기존 LRM이 마지막 턴만 think하도록 학습됨

```python
# 기존 포맷 (유지)
턴1: action1 → env1
턴2: action2 → env2
턴3: think → action3 → reward  # 마지막만 think
```

**이유**:
- 적은 데이터로 포맷 변경은 위험
- Distribution shift로 기존 능력 손상 가능
- 포맷 유지하면서 step 기반으로 이전 action 학습 가능

**학습되는 것**:
- action1, action2: step 샘플로 학습
- think3, action3: 마지막 턴 샘플로 학습

### 2.3 Advantage: Mean Centering (std 안 나눔)

**Z-score vs Mean centering 비교**

```python
# 극단 케이스: 성공 1개, 실패 99개
mean = 0.01, std = 0.099

# Mean centering
advantage(성공) = +0.99
advantage(실패) = -0.01

# Z-score
advantage(성공) = +10.0  # 폭발!
advantage(실패) = -0.1
```

**Mean centering 선택 이유**:
- Z-score는 희소 샘플을 극단적으로 증폭
- 하나의 샘플이 gradient를 지배할 위험
- Binary reward (0,1)에서 std 나누기는 스케일링일 뿐
- Rejection sampling으로 극단 배치 방지 → mean centering으로 충분

### 2.4 Value Function: 사용 안 함

**GRPO 방식 채택 이유**:
- Value function 학습 비용 절감
- 같은 태스크에서 multiple rollout → 상대 비교로 baseline 대체
- DeepSeek이 검증한 방법

**Credit assignment 해결**:
- Step 단위 샘플링으로 턴별 학습
- 충분한 샘플 수로 자연스러운 상쇄/강화

### 2.5 Gamma, Lambda: 둘 다 1.0

```python
gamma = 1.0   # 디스카운트 없음
lambda = 1.0  # Monte Carlo return
```

**이유**:
- Sparse reward (최종 0/1) → 중간 신호 없음
- 알파고도 승패로만 학습, 잘 작동함
- 최종 결과로 전체 trajectory 평가

---

## 3. Step 기반 샘플링 상세

### 3.1 샘플 구성

```python
# 성공 trajectory (3턴)
A → B → (think + C) → reward=1

# 생성되는 샘플
(state0, A) → return=1
(state1, B) → return=1
(state2, think+C) → return=1

# 실패 trajectory (5턴)
A → B → X → Y → (think + Z) → reward=0

# 생성되는 샘플
(state0, A) → return=0
(state1, B) → return=0
(state2, X) → return=0
(state3, Y) → return=0
(state4, think+Z) → return=0
```

### 3.2 Advantage 계산 예시

```python
# 배치: 성공 3턴 + 실패 5턴 = 8개 샘플
returns = [1, 1, 1, 0, 0, 0, 0, 0]
mean = 3/8 = 0.375

advantages:
  성공 샘플: 1 - 0.375 = +0.625
  실패 샘플: 0 - 0.375 = -0.375

# 공통 action (A, B)의 총 gradient
A: +0.625 + (-0.375) = +0.25  # 순 강화
B: +0.625 + (-0.375) = +0.25  # 순 강화
C: +0.625
X, Y, Z: -0.375 each
```

### 3.3 왜 이게 합리적인가

**짧은 성공 trajectory의 action이 더 강화되는 이유**:
- 효율적으로 문제 해결 = 좋은 것
- 샘플 수가 적음 → mean이 성공 쪽으로 덜 쏠림 → 상대적 advantage 높음

**짧은 실패 trajectory의 action이 더 약화되는 이유**:
- 빠른 포기/잘못된 판단 = 나쁜 것
- 조기 포기보다 끝까지 시도하는 게 나음

---

## 4. Rejection Sampling

```python
# 배치 내 성공/실패가 섞여야 함
if all(rewards == 1) or all(rewards == 0):
    reject_batch()

# 이유
# - 전부 성공: advantage가 전부 0에 가까움 (mean ≈ 1)
# - 전부 실패: advantage가 전부 0에 가까움 (mean ≈ 0)
# - 학습 신호 없음
```

---

## 5. 전체 학습 파이프라인

```python
for epoch in epochs:
    # 1. Rollout 수집
    trajectories = collect_rollouts(tasks, n_per_task=8)

    # 2. Step 단위 샘플로 변환
    samples = []
    for traj in trajectories:
        for t, (state, action) in enumerate(traj.steps):
            samples.append({
                'state': state,
                'action': action,  # 마지막 턴이면 think+action
                'return': traj.reward
            })

    # 3. 배치 구성 (mixed success/fail)
    batches = create_mixed_batches(samples)

    # 4. Advantage 계산 및 업데이트
    for batch in batches:
        returns = [s['return'] for s in batch]
        mean_r = mean(returns)

        for sample in batch:
            advantage = sample['return'] - mean_r
            update_policy(sample['state'], sample['action'], advantage)
```

---

## 6. 요약

| 항목 | 결정 | 핵심 근거 |
|------|------|----------|
| Sample 단위 | Step | 이전 action 학습 + 합리적 길이 bias |
| Think 포맷 | 기존 유지 | 적은 데이터로 포맷 변경 위험 |
| Advantage | Mean centering | Z-score는 극단 케이스에서 불안정 |
| Value function | 없음 | GRPO 방식, 비용 절감 |
| Gamma | 1.0 | Sparse reward, 알파고 방식 |
| Lambda | 1.0 | Monte Carlo return |
| Rejection | Mixed batch | 학습 신호 확보 |
