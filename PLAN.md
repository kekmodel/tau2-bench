# tau2-bench SLIME 통합 개선 계획

## 현재 상태 분석

### 기존 tau2-bench 구현 (`slime/examples/tau2-bench/`)
- `generate_with_gym.py`: 기본 구조만 존재, 실제 동작 미완성
- `config.py`: 환경 변수 기반 설정 (양호)
- `test_full_sample.py`: CPU 테스트용 모의 구현

### 주요 문제점
1. **Tool Parsing 부재**: LLM 응답에서 tool call을 파싱하는 로직 없음
2. **Chat Template 미사용**: 단순 문자열 포맷팅 사용 (모델 호환성 문제)
3. **Message 관리 미흡**: tau2 메시지 타입과의 변환 로직 부족
4. **Token Delta 미구현**: 멀티턴 토큰/loss_mask 계산 누락

---

## 구현 계획

### Phase 1: 핵심 유틸리티 모듈 작성

#### 1.1 `tool_parser.py` - Tool Call 파싱 모듈

**목적**: sglang 응답에서 tool call을 파싱하여 tau2 호환 형식으로 변환

**구현 내용**:
```python
# 필요한 기능
- parse_tool_calls(response: str, tools_info: list) -> ParseResult
- convert_to_tau2_message(parsed: ParseResult) -> AssistantMessage
- validate_tool_call(tool_call, available_tools) -> bool
```

**참조**: `slime/examples/tau-bench/sglang_tool_parser.py`, `openai_tool_adapter.py`

**핵심 로직**:
1. sglang의 `FunctionCallParser` 사용 (qwen25 parser)
2. 파싱 결과를 tau2의 `AssistantMessage`/`ToolCall` 타입으로 변환
3. 파싱 실패 시 plain text로 fallback

---

#### 1.2 `message_utils.py` - 메시지 변환 유틸리티

**목적**: tau2 메시지 ↔ chat template 메시지 변환

**구현 내용**:
```python
# tau2 Message -> dict (for chat template)
def tau2_message_to_dict(msg: Message) -> dict[str, Any]

# dict -> tau2 Message
def dict_to_tau2_message(msg_dict: dict) -> Message

# Tool spec 변환
def build_tool_specs(tools: list[Tool]) -> list[dict]

# Token delta 계산 (from tau-bench trainable_agents.py)
def get_token_delta(tokenizer, messages, tools) -> tuple[list[int], list[int]]
```

---

#### 1.3 `config.py` 확장

**추가 설정**:
```python
class SlimeConfig:
    # 기존 설정 유지
    domain: str
    task_split: str
    user_llm: str
    max_steps: int

    # 추가 설정
    tool_parser_type: str = "qwen25"  # qwen25, llama3, etc.
    model_type: str = "qwen3"  # 모델별 chat template 구분
    return_logprob: bool = False  # TIS metrics용
    max_response_tokens: int = 1024
```

---

### Phase 2: 메인 생성 함수 재작성

#### 2.1 `generate_with_gym.py` 전면 재작성

**구조**:
```python
async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    SLIME Sample 생성 - tau2 AgentGymEnv 사용

    Flow:
    1. 환경 초기화 (AgentGymEnv)
    2. 초기 프롬프트 구성 (system + user message)
    3. Chat template 적용 및 토큰화
    4. Multi-turn interaction loop:
       a. sglang 호출
       b. Tool call 파싱
       c. tau2 메시지 변환
       d. 환경 step
       e. Token delta 및 loss_mask 업데이트
    5. 최종 Sample 구성
    """
```

**핵심 변경사항**:

1. **Chat Template 사용**:
```python
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tool_specs,
)
```

2. **Tool Call 파싱 통합**:
```python
parsed = tool_parser.parse_tool_calls(response, tool_specs)
if parsed.success:
    action_msg = convert_to_tau2_message(parsed)
else:
    # Plain text fallback
    action_msg = AssistantMessage(content=response)
```

3. **정확한 Token Delta 계산**:
```python
# Assistant 응답
token_ids, mask = get_token_delta(tokenizer, messages, tool_specs)
response_token_ids.extend(token_ids)
loss_masks.extend(mask)  # [1, 1, 1, ...]

# 환경 응답 (user/tool)
token_ids, mask = get_token_delta(tokenizer, messages, tool_specs)
response_token_ids.extend(token_ids)
loss_masks.extend(mask)  # [0, 0, 0, ...]
```

---

### Phase 3: 데이터 준비 스크립트

#### 3.1 `prepare_data.py` - 태스크 데이터 생성

**목적**: tau2 태스크를 SLIME 입력 형식으로 변환

**출력 형식** (JSONL):
```json
{"index": "task_id", "metadata": {"domain": "telecom", "task_split": "train"}}
```

**구현**:
```python
def prepare_tasks(domain: str, task_split: str, output_path: str):
    """
    tau2 태스크를 SLIME 입력 JSONL로 변환
    """
    tasks = registry.get_tasks_loader(domain)(task_split)
    with open(output_path, 'w') as f:
        for task in tasks:
            row = {
                "index": task.id,
                "metadata": {
                    "domain": domain,
                    "task_split": task_split,
                }
            }
            f.write(json.dumps(row) + "\n")
```

---

### Phase 4: 실행 스크립트 및 테스트

#### 4.1 `run_qwen3_4B.sh` - 학습 실행 스크립트

**핵심 설정**:
```bash
ROLLOUT_ARGS=(
   --prompt-data /path/to/tau2_tasks.jsonl
   --input-key index
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_gym.generate
)

# 환경 변수
export TAU2_DOMAIN=telecom
export TAU2_TASK_SPLIT=train
export TAU2_USER_LLM=gpt-4.1
```

---

#### 4.2 `test_integration.py` - 통합 테스트

**테스트 항목**:
1. Tool parsing 정확성
2. Message 변환 무결성
3. Token/loss_mask 정합성
4. 전체 에피소드 실행
5. Reward 계산 검증

```python
async def test_full_episode():
    """실제 sglang 서버 없이 모의 테스트"""

async def test_tool_parsing():
    """다양한 tool call 형식 파싱 테스트"""

async def test_token_alignment():
    """토큰과 loss_mask 정렬 검증"""
```

---

### Phase 5: 추가 기능 (선택)

#### 5.1 Log Probability 수집
```python
if config.return_logprob:
    payload["return_logprob"] = True
    # output_token_logprobs에서 추출
    sample.rollout_log_probs = [...]
```

#### 5.2 평가 모드 분리
```python
async def generate_eval(args, sample, sampling_params) -> Sample:
    """평가용 생성 (temperature=0 등)"""
```

---

## 파일 구조

```
slime/examples/tau2-bench/
├── __init__.py
├── config.py              # 설정 모듈 (확장)
├── generate_with_gym.py   # 메인 생성 함수 (재작성)
├── tool_parser.py         # Tool call 파싱 (신규)
├── message_utils.py       # 메시지 변환 유틸 (신규)
├── prepare_data.py        # 데이터 준비 스크립트 (신규)
├── run_qwen3_4B.sh        # 실행 스크립트 (신규)
├── test_integration.py    # 통합 테스트 (신규)
└── README.md              # 문서 (신규)
```

---

## 구현 우선순위

| 순서 | 모듈 | 중요도 | 예상 작업량 |
|------|------|--------|------------|
| 1 | `tool_parser.py` | 필수 | 중 |
| 2 | `message_utils.py` | 필수 | 중 |
| 3 | `generate_with_gym.py` 재작성 | 필수 | 대 |
| 4 | `config.py` 확장 | 필수 | 소 |
| 5 | `prepare_data.py` | 필수 | 소 |
| 6 | `run_qwen3_4B.sh` | 필수 | 소 |
| 7 | `test_integration.py` | 권장 | 중 |
| 8 | `README.md` | 권장 | 소 |

---

## 의존성 요구사항

```
# 기존
tau2  # tau2-bench 패키지
slime  # SLIME 프레임워크
transformers  # 토크나이저

# 추가 필요
sglang  # FunctionCallParser 사용
```

---

## 검증 체크리스트

- [ ] Tool call 파싱이 다양한 형식에서 동작
- [ ] Chat template이 모델에 맞게 적용됨
- [ ] Token과 loss_mask 길이가 일치
- [ ] Assistant 토큰만 loss_mask=1
- [ ] tau2 환경 step이 올바른 action으로 호출됨
- [ ] Reward가 tau2 평가기에서 정상 계산됨
- [ ] SLIME 학습 루프에서 Sample이 올바르게 소비됨
