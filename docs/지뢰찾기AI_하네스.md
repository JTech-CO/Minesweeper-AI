# 지뢰찾기 AI — Claude Code 작업 하네스 (Harness)

**버전**: 0.1
**작성일**: 2026년 06월 04일
**관계 문서**: 루트 `CLAUDE.md`(전역 규칙), `PROGRESS.md`(상태 인계), `docs/지뢰찾기AI_기술백서.md`·`디자인백서.md`·`파일트리.md`·`로컬환경_4060.md`

> 이 문서는 "무엇을 만드는가"(백서)가 아니라 **"어떻게 진행·검증·복구하는가"(운영 규율)** 를 정의한다. 각 phase의 "완료"는 코드가 도는 것이 아니라 **측정 가능한 게이트(DoD) 통과**다. RL 특성상 "에러 없이 도는데 학습은 안 됨"이 흔하므로, 게이트로 막지 않으면 잘못된 완료 판정이 다음 phase로 전파된다.

---

## 0. 사용법

### 0.1 세션 루프
1. **시작**: `PROGRESS.md` 읽기 → 현재 phase·다음 할 일·미결 질문 확인 → 본 문서의 해당 phase 절 + 백서 참조 절 확인.
2. **작업**: 한 번에 한 phase. 각 작업 단위마다 검증 명령을 돌려 게이트로 확인.
3. **종료**: `PROGRESS.md` 갱신(완료 체크·다음 할 일·새 미결 질문·결정 로그) → 커밋.

### 0.2 phase 완료 판정
- DoD 항목이 **전부** 충족되어야 완료. 하나라도 미달이면 미완료로 두고 다음 phase로 가지 않는다.
- 게이트를 못 넘는데 우회 충동이 들면 → §3 멈춤 규칙.

### 0.3 의존 순서
M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10.
일부는 병렬 가능하나(M3는 M2와 병렬 가능 등), 위 순서를 기본으로 한다. 각 phase의 "진입조건"이 선행 게이트를 명시한다.

---

## 1. Phase별 진입조건 · 할 일 · DoD · 검증

### M1 — `@msai/core` 엔진
- **진입조건**: 레포 스캐폴딩(루트 설정, pnpm workspace, tsconfig.base, eslint 경계 룰).
- **할 일**: `types.ts` → `rng.ts`(시드) → `board.ts`(생성·첫클릭안전·BFS 캐스케이드·승패) → `difficulty.ts` → `metrics.ts`(3BV/3BV·s) → `env.ts`(reset/step/legalActionMask/render). vitest 테스트.
- **참조**: 기술 백서 §3, §4.3.
- **DoD**:
  1. `pnpm --filter @msai/core test` 전부 그린.
  2. 결정론: 동일 시드 → 동일 보드(테스트로 고정).
  3. 첫 클릭이 지뢰가 아님 보장.
  4. 0 셀 오픈 시 캐스케이드가 정확(인접 0 영역 전부 오픈).
  5. 3BV가 알려진 소형 케이스와 일치.
- **검증**: `pnpm --filter @msai/core test`

### M2 — `@msai/core` 논리 솔버
- **진입조건**: M1 통과.
- **할 일**: `solver/single-point.ts` → `patterns.ts`(1-2-1, 1-2-2-1) → `csp.ts`(프론티어 열거) → `probability.ts` → `index.ts`(통합). NG 보드 생성기(reject-sampling: 솔버로 "추측 없이 풀림"을 확인하며 생성).
- **참조**: 기술 백서 §4.3.
- **DoD**:
  1. **거짓양성 0**(불변식): 솔버가 "확실 안전"이라 한 셀이 실제 지뢰인 경우가 절대 없음. 대량 무작위 보드로 검증.
  2. NG 보드(생성기로 만든)에 대해 솔버 단독 클리어율 ~100%.
  3. CSP 확률이 손계산 가능한 케이스(1-2-1, 코너 등)와 일치.
- **검증**: `pnpm --filter @msai/core test` (solver.test.ts)
- **주의**: 거짓양성이 1건이라도 나오면 솔버는 신뢰 불가 → 통과 금지.

### M3 — `server/app` 골격 (FastAPI + DB)
- **진입조건**: M2 통과(권장). 로컬환경 가이드대로 `docker compose up -d db` 가능.
- **할 일**: FastAPI 앱, `core/config.py`(.env), `db/`(SQLAlchemy 세션·ORM: TrainingRun/Model/BenchmarkResult), Alembic init + 최초 마이그레이션, `schemas/`(pydantic), `GET /health`, 모델 메타 list/get/put(트레이너 의존부는 스텁 가능).
- **참조**: 기술 백서 §2.3, §3.1; 로컬환경 가이드.
- **DoD**:
  1. `docker compose up -d db` 후 `alembic upgrade head` 성공(테이블 생성).
  2. `uvicorn app.main:app` 기동, `GET /health` 200.
  3. 모델 메타 CRUD 왕복(가짜 레코드 list/get/put).
  4. pydantic 스키마가 TS `@msai/core` 타입과 모순 없음(수동 점검 또는 생성).
- **검증**: `alembic upgrade head` → `curl localhost:8000/health` → `pytest`

### M4 — `server/trainer` 학습 핵심 ★
- **진입조건**: M3 통과 + **CUDA torch 검증**(`torch.cuda.is_available()` True, 로컬환경 가이드 §3).
- **할 일**: `env.py`(core 규칙 1:1 재현) → `encoding.py` → `model.py`(fully-conv CNN, 선택 dueling) → `replay.py`(PER) → `dqn.py`(Double DQN, target net, n-step) → `reward.py` → `train.py`(에피소드 루프·체크포인트). `tests/test_env.py`.
- **참조**: 기술 백서 §4, §5, §6.
- **DoD** (코드 실행이 아니라 **학습됨**을 본다):
  1. Beginner(9×9·10)에서 **평가 승률이 명확히 우상향**해 기술 백서 커리큘럼 게이트(예 0.85)에 도달.
  2. 학습 loss가 NaN/발산하지 않음.
  3. 체크포인트(.pt) 저장 후 재로드 시 동일 추론 재현.
  4. `test_env.py`: 파이썬 env가 동일 시드·동일 행동열에서 결정론적이고 규칙이 올바름.
- **검증**: `python -m trainer.train --difficulty beginner` 후 평가 승률 로그/곡선 + `pytest server/tests/test_env.py`
- **주의**: 승률 게이트 미달이면 **미완료**. loss만 보고 완료 판정 금지. 평평하면 §2 런북 "학습 평평".

### M5 — 메트릭 스트림 + 체크포인트 영속화
- **진입조건**: M4 통과.
- **할 일**: `ws/metrics.py`(배치/스로틀), `train.py`에서 메트릭 콜백 → WS push, 체크포인트 생성 시 `Model` DB 레코드 + `storage/weights.py`로 파일 저장.
- **참조**: 기술 백서 §2.5, §7.
- **DoD**:
  1. WS 클라이언트(wscat/간단 테스트)로 학습 메트릭(에피소드·승률·loss·ε) 수신.
  2. 체크포인트 생성 시 `Model` row 생성 + 파일이 `MODEL_STORAGE_DIR`에 저장.
- **검증**: `wscat -c ws://localhost:8000/ws/metrics` 수신 확인 + DB 조회

### M6 — 커리큘럼 학습
- **진입조건**: M4·M5 통과.
- **할 일**: `curriculum.py`(단계별 승률 게이트, 가중치 승계, ε 재상승), `train.py` 연동.
- **참조**: 기술 백서 §6.2.
- **DoD**:
  1. Beginner→Intermediate→Expert 자동 승급 동작(각 단계 게이트 통과 후 전이, 가중치 유지).
  2. Expert에서 학습 곡선이 **~40% 근방으로 수렴**.
- **검증**: 커리큘럼 전이 로그 + Expert 승률 추이
- **주의**: Expert 승률이 40%대에서 안 오르는 것은 **정상 상한**(강제 50/50). "실패"로 보고 무한 튜닝하지 말 것(§2 런북 6).

### M7 — ONNX export + `@msai/agent` 추론 ★
- **진입조건**: M4+ (학습된 모델 존재).
- **할 일**: `trainer/export.py`(`torch.onnx.export`) → `agent/encoding.ts`(파이썬과 1:1) → `agent/onnx-session.ts`(onnxruntime-web) → `agent/inference/{policy,hybrid}.ts` → `server/tests/test_encoding_parity.py`.
- **참조**: 기술 백서 §4.1, §4.3, §7.
- **DoD** (불변식):
  1. **인코딩 패리티**: 고정 보드 입력 셋에 대해 `encoding.ts` 출력 == `encoding.py` 출력(완전 일치 또는 부동소수 오차 내).
  2. **ONNX 추론 패리티**: 동일 입력에 대해 onnxruntime-web 출력 == PyTorch 출력(오차 ≤ 1e-3).
  3. 하이브리드(`hybrid.ts`)가 NG 보드를 ~100% 클리어.
- **검증**: `pytest server/tests/test_encoding_parity.py` + ONNX↔torch 수치 비교 스크립트 + 브라우저/노드에서 하이브리드 NG 시연
- **주의**: 이게 깨지면 학습한 모델이 브라우저에서 엉뚱하게 동작한다. **절대 우회 금지**, 안 맞으면 §2 런북 2·3.

### M8 — `@msai/web` 대시보드 (React)
- **진입조건**: M3(API)·M5(WS)·M7(추론) 통과.
- **할 일**: React 앱(`main.tsx`·`App.tsx`), `views/{Train,Play,Benchmark,Models}`, `components/{layout,ui,features}`, TanStack Query·Zustand, WS 구독, BoardView·MetricChart, `styles/tokens.css`.
- **참조**: 디자인 백서 전체; 기술 백서 §4.1, §5.
- **DoD**:
  1. 서버 연결 시 모델 목록 표시, 학습 시작 → 실시간 메트릭 차트 갱신.
  2. Play에서 ONNX+솔버 하이브리드로 한 판 진행 시각화(확실/추측 셀 색 구분).
  3. 디자인 토큰·anti-cliché 준수.
  4. 가로 스크롤 없음 + 기본 접근성(키보드·aria·색 외 표식).
- **검증**: `pnpm --filter @msai/web build` 그린 + 수동 시연(학습→차트, 플레이 한 판)

### M9 — `@msai/online-adapter` (Chrome MV3)
- **진입조건**: M7(추론) 통과 + 실제 사이트 접근 가능.
- **할 일**: `content/dom-contract.ts`(DevTools 검증) → `read-board.ts`(DOM→core Board) → `dispatch.ts`(클릭/플래그/코드) → `loop.ts` → `manifest.json` → `popup`. `online-adapter/CLAUDE.md` 가드레일.
- **참조**: 기술 백서 §7(하이브리드); 파일 트리 §3.4; (사이트 DOM은 실측).
- **DoD**:
  1. 실제 minesweeper.online 보드를 정확히 파싱(셀 상태를 화면과 수동 대조해 일치).
  2. 클릭 디스패치가 실제로 셀을 오픈/플래그.
  3. **NG 모드에서 한 판 자동 클리어** 시연.
  4. 가드레일 준수(스크래핑 금지·레이트 제한·권한 최소).
- **검증**: 빌드 후 사이드로드 → 실제 사이트 NG 한 판 시연 + DOM 파싱 대조
- **주의**: DOM이 안 맞으면 **`dom-contract.ts`만** 갱신(다른 파일 수정 금지). §2 런북 10.

### M10 — 벤치마크 · (선택)리더보드 · 배포
- **진입조건**: M8 통과.
- **할 일**: `api/benchmark.py`(N판 시행·결과 저장), `Benchmark.tsx` 표시, (선택)leaderboard, `.github/workflows/{ci,deploy}.yml`, 서버 이미지·web 정적·확장 zip.
- **참조**: 기술 백서 §2.3·§2.4; 파일 트리 §1.
- **DoD**:
  1. 벤치 N판 → `BenchmarkResult` 저장 → 대시보드 표시(승률·평균 시간·3BV·s).
  2. CI 그린(JS lint/typecheck/test/build + Python ruff/pytest + 확장 빌드).
  3. 배포 산출물 생성(web 정적·서버 이미지·확장 zip).
- **검증**: CI 통과 + 벤치 리포트 + 아티팩트 확인

---

## 2. 지뢰밭 런북 (증상 → 원인 → 조치)

| # | 증상 | 흔한 원인 | 조치 |
|---|---|---|---|
| 1 | `torch.cuda.is_available()` == False | CPU 휠 설치 / 드라이버↔CUDA 불일치 / 잘못된 index-url | 로컬환경 가이드 §2로 재설치(cuXXX index-url), `nvidia-smi`로 드라이버 확인 |
| 2 | 인코딩 패리티 실패 | TS↔Python 채널 순서/개수 불일치, 정규화 값 차이, 행/열 우선 차이, hidden/flag 인코딩 차이 | 기술 백서 §4.1 채널표를 단일 기준으로 양쪽 정렬. 채널별 출력 비교로 어긋난 채널 특정 |
| 3 | ONNX↔torch 추론 불일치 | opset 버전, 입력 레이아웃(NCHW/NHWC), `eval()`/`no_grad` 누락, dynamic axes, 마스킹 적용 위치 불일치 | export 시 `model.eval()`, opset·입력 shape 명시, 마스킹은 **모델 외부**에서 일관 적용, dummy input 점검 |
| 4 | 학습 승률 평평(loss는 정상) | **행동 마스킹 누락**(이미 오픈 셀 선택), 보상 신호 부재/과도, 리플레이 워밍업 전 학습, ε 스케줄·lr 부적절, target 갱신 주기 | 마스킹부터 확인(가장 흔함) → 보상 스케일 → 리플레이 충분히 찬 뒤 업데이트 → ε/lr 조정 → 5×5 같은 초소형 보드 과적합(sanity overfit) 확인 |
| 5 | loss NaN/발산 | lr 과대, 보상 스케일 과대, gradient 폭주, target 식 오류 | lr 낮춤, gradient clipping, 보상 정규화, Double DQN target 식 점검 |
| 6 | Expert 승률 ~40%에서 정체 | **정상**(강제 50/50 상한) | 실패 아님. 상한 인지. 일관 시연은 NG 모드. 무한 튜닝 금지 |
| 7 | `alembic upgrade head` 실패 | DB 미기동, DATABASE_URL 불일치, autogenerate 시 ORM 모델 import 누락 | `docker compose up -d db` 확인, `.env` DATABASE_URL 점검, alembic env.py에 ORM 모델 import |
| 8 | CORS 에러(web→server) | CORS_ORIGINS에 web/확장 출처 누락 | `.env` CORS_ORIGINS에 `http://localhost:5173`, `chrome-extension://*` 추가 |
| 9 | WebSocket 연결 실패 | vite proxy/경로, 서버 WS 라우트, 방화벽 | vite proxy 설정, ws 경로 일치, 서버 로그 확인 |
| 10 | 확장이 보드 오독 / 클릭 안 먹음 | 사이트 DOM 변경(셀렉터·클래스), 줌 size 클래스 변동, 이벤트 타입 불일치 | **`dom-contract.ts`만** DevTools로 갱신, 클릭은 셀 엘리먼트 대상 합성 이벤트(mousedown/up/contextmenu), 클릭 간 지연 조정 |
| 11 | GPU util 낮고 학습 느림 | 환경 스텝이 CPU 바운드 직렬 | 환경 벡터화/병렬 워커, 배치 추론(기술 백서 §2.5) |
| 12 | 경계 위반 lint 에러 | `agent`↔`online-adapter` 상호 import, `core`에서 onnx/DOM import | 경계 준수(파일 트리 §2), 공유는 `core` 타입/솔버만 |

> 런북에 없는 새 이슈는 임시 처치 후 `PROGRESS.md` 결정 로그에 기록하고, 반복되면 이 표에 추가한다.

---

## 3. 멈춤 규칙 (STOP) 상세

### 3.1 멈춰야 하는 상황
- 같은 에러/테스트 실패를 **서로 다른 방법으로 3회** 시도해도 미해결.
- DoD 게이트(특히 M4 승률·M7 패리티·M2 거짓양성)를 못 넘는데 우회 충동.
- 불변식(루트 `CLAUDE.md` §5)을 깨야만 통과 가능.
- 큰 아키텍처 변경 필요 추정(예: 패리티가 어려워 추론을 서버로 이전).
- 외부 제약(GPU 없음, 사이트 DOM 접근 불가, 약관 위반 소지).

### 3.2 멈출 때 절차
1. `PROGRESS.md`에 기록: **증상 / 재현 방법 / 시도한 것들 / 가설 / 막힌 지점**.
2. 사용자에게 위 요약을 보고하고, 선택지가 있으면 제시한 뒤 결정을 요청한다.
3. 결정 전까지 **불변식을 깨는 임시 우회를 만들지 않는다.**

### 3.3 절대 금지
- 테스트 삭제·약화 또는 게이트 수치 임의 하향으로 "통과" 위장.
- 패리티가 깨진 채 다음 phase 진행.
- 아키텍처·패키지 경계 임의 변경(사용자 승인 없이).
- 가중치(`*.pt`/`*.onnx`)·시크릿(`.env`) 커밋.
- 가드레일 약화(스크래핑·과도 요청·과한 권한).

---

## 4. 검증 우선순위 (한 줄 요약)
규칙 정확성(M2 거짓양성 0) > 학습 실효(M4 승률) > 이식 정합성(M7 패리티) > UX/배포(M8·M10). 앞 단계의 게이트가 깨지면 뒤 단계 작업은 의미가 없으므로, 항상 앞에서부터 굳힌다.
