# PROGRESS.md — 진행 상태 (세션 인계용)

> Claude Code는 세션 간 상태가 리셋된다. **시작 시 이 파일을 먼저 읽고, 종료 시 갱신**한다(루트 `CLAUDE.md` §1). 절차·DoD는 `docs/지뢰찾기AI_하네스.md` 참조.

---

## 현재 상태
- **현재 Phase**: M3 **완료** → 다음 M4 (trainer ★, 학습 핵심)
- **마지막 갱신**: 2026-06-04
- **환경 점검**: Node 25 / pnpm 11.1.3 / git 2.54. 서버 venv=**Python 3.12.13**(uv 관리, `server/.venv`). GPU=**RTX 4060 Laptop, driver 556.12, 8GB**. DB = dev SQLite로 `alembic upgrade head` 성공. **torch.cuda.is_available() = [x] True**(torch 2.6.0+cu124, GPU matmul 동작). numpy 2.4.6. → **M4 진입조건 충족.**

---

## Phase 체크리스트
각 phase는 `하네스` §1의 DoD를 **전부** 충족해야 [x].

- [x] **M1** core 엔진 — vitest 그린(32/32)·결정론·첫클릭안전·캐스케이드·3BV
- [x] **M2** core 솔버 — 거짓양성 0(420+스윕2150보드)·NG 솔버 ~100%(생성10판)·CSP 확률 일치
- [x] **M3** server 골격 — alembic head(SQLite)·uvicorn /health 200·모델 메타 CRUD·pydantic↔TS 무모순·ruff/pytest 그린
- [ ] **M4** trainer — Beginner 승률 게이트 도달·loss 비발산·체크포인트 재현
- [ ] **M5** 메트릭 — WS 수신·체크포인트 DB 기록
- [ ] **M6** 커리큘럼 — 자동 승급·Expert ~40% 수렴
- [ ] **M7** ONNX+agent — 인코딩 패리티·ONNX↔torch 일치·NG 하이브리드 ~100%
- [ ] **M8** web — 실시간 차트·플레이 시각화·토큰/anti-cliché·접근성
- [ ] **M9** adapter — 보드 파싱 일치·클릭 동작·NG 한 판 클리어
- [ ] **M10** 배포 — 벤치 저장/표시·CI 그린·배포 산출물

---

## 다음 할 일 (구체적으로) — M4 trainer ★ (학습 핵심) · **진행 중**
> 진입조건 충족: `torch.cuda.is_available()` True(2.6.0+cu124, RTX 4060). 환경 셋업·`env.py` 완료.
- [x] 0. 환경: Python 3.12 venv + CUDA torch + numpy (커밋 `9db0dd2`).
- [x] 1. `trainer/env.py` — board.ts 규칙 1:1 재현 + `tests/test_env.py` 12 그린 (커밋 `58839f6`). **DoD #4 충족.**
- [x] 2. `trainer/encoding.py` — (11,H,W) 채널: hidden + 숫자0~8 원핫 + 지뢰밀도. **M7 패리티 계약**(`agent/encoding.ts`와 동일해야 함, docstring에 명시).
- [x] 3. `trainer/model.py`(fully-conv residual QNet, no BN) · `replay.py`(PER sum-tree) · `dqn.py`(Double DQN·n-step·target, **외부 마스킹+유한 MASK_VALUE로 terminal NaN 방지**) · `reward.py`.
- [x] 4. `trainer/train.py`(ε/β 스케줄·`--train-freq`·체크포인트·게이트 조기종료) · `evaluate.py`. `tests/test_trainer_shapes.py` 5 그린.
- [~] 5. **sanity(5×5×3) 학습 확인**: eval 0.32→0.66 단조상승, loss 안정. **체크포인트 재로드 추론 재현(0.66→0.66)**. → **DoD #2·#3 충족.** **Beginner 학습은 background 진행 중**(아래).
- **DoD #1만 남음**: Beginner eval 승률 게이트(0.85) 도달.

### Beginner 백그라운드 학습 (진행 중)
- 실행: `python -m trainer.train --difficulty beginner --train-freq 4 --gate 0.85 --tag beginner ...` (PID는 `server/storage/checkpoints/beginner.pid`).
- 로그: `server/storage/checkpoints/beginner.out`(stdout) / `beginner.err`(stderr) / `beginner_metrics.csv`(eval 추이). 체크포인트: `beginner_best.pt`.
- 모니터: `Get-Content beginner.out -Tail 20` 또는 `beginner_metrics.csv` 확인. 첫 eval=ep 5000. 게이트 도달 시 자동 종료.
- **웹 대시보드**(브라우저 자동): `python -m trainer.dashboard --tag beginner` → http://127.0.0.1:8800. 서버가 모델로 플레이→**WebSocket으로 한 수씩 스트리밍**, 브라우저가 보드/클릭시퀀스/풀이시간/승패 + 승률곡선·loss·ε 렌더(`dashboard.html`, 디자인 토큰·anti-cliché). 터미널 뷰어 `trainer.watch`도 동일 정보 제공. (클라이언트 ONNX 추론 풀 React 대시보드는 M8.)
- 멈출 때: `Stop-Process -Id <pid>`.
> 학습 평평하면 런북 4(마스킹은 검증됨). 0.85 미도달·정체 시 하이퍼파라미터(lr·eps decay·width) 튜닝은 후속.

---

## 미결 질문 / 블로커 (사용자 결정 필요)
> 멈춤 규칙(하네스 §3)으로 멈췄을 때 여기에 적는다: 증상 / 재현 방법 / 시도한 것 / 가설 / 필요한 결정.

- **[해결됨] M3 DB 환경** (2026-06-04): 사용자가 **개발용 SQLite 승인**. 포터블 타입으로 작성해 Postgres 호환 유지, `alembic upgrade head` 성공. docker-compose+PG는 CI/운영용으로 작성. (결정 로그 참조)
- **[해결됨] M4 PyTorch ↔ Python 버전** (2026-06-04): 사용자 승인하에 uv로 Python **3.12.13** 설치 → `server/.venv` 재생성 → 의존성+**torch 2.6.0+cu124** 설치 → `torch.cuda.is_available()` **True**(RTX 4060). M3 코드도 3.12에서 회귀 그린(pytest 3, alembic check). (현재 미결 블로커 없음.)

---

## 결정 로그 (Decision Log)
> 백서에서 벗어났거나 명확히 한 결정, 런북에 없던 이슈의 처치를 날짜와 함께 기록. 나중에 백서/하네스에 반영.

| 날짜 | 결정/이슈 | 근거 | 영향 문서 |
|---|---|---|---|
| 2026-06-04 | **첫클릭 정책 기본값 = `safe-area`**(클릭셀+8이웃 지뢰 제외, 공간 부족 시 `safe-cell`로 폴백) | 첫 클릭이 항상 0-셀→오프닝, minesweeper.online 기본 동작과 일치 | **M4 `env.py`가 동일 규칙 재현 필수**, M9 |
| 2026-06-04 | 보드 표현 = flat row-major typed array(`index=row*cols+col`), 지뢰는 **첫 reveal 시 배치** | 결정론·할당 경량(롤아웃), Python env와 매핑 단순 | M4 `env.py` 동일 레이아웃 |
| 2026-06-04 | RNG = `mulberry32`(시드), 부분 Fisher–Yates로 지뢰 선택 | 재현성·Python 재구현 용이 | M4(필요 시 동일 RNG) |
| 2026-06-04 | TS env 보상값(`DEFAULT_REWARD`)은 **편의 미러**일 뿐, 학습 보상 정본은 M4 `reward.py` | 규칙(룰)은 패리티 대상이나 보상 셰이핑은 트레이너 소관 | M4 `reward.py` |
| 2026-06-04 | 툴체인 설치본: TypeScript ^6.0, ESLint ^10.4, vitest ^4.1, typescript-eslint ^8.60, prettier ^3.8 (pnpm 최신 해석) | 신규 레포·최신 안정 | — |
| 2026-06-04 | **솔버 가시정보 강제**: `solver/view.ts`(파일트리에 없는 내부 베이스) 추가. `toSolverView`가 숨은 셀 `adjacent`를 `-1`로 마스킹(엔진은 전 셀 adjacency를 채우므로 그대로 주면 컨닝) | 거짓양성 0 불변식을 구조적으로 보장 | 파일트리(솔버 모듈에 view.ts 추가) |
| 2026-06-04 | 부분집합 규칙 = 일반 subset rule(1-2-1/1-2-2-1 포함). CSP는 연결성분 분리 + 완전열거, `COMPONENT_CAP=20` 초과 성분은 건너뜀(추측 위임, 건전성 유지) | 백서 "부분집합 규칙"·프론티어 열거 | 기술백서 §4.3 |
| 2026-06-04 | **버그수정(중요)**: `cspCertain` 바다-경계에서 프론티어 지뢰 이중 계상 → 바다를 안전 오판정(거짓양성). `remaining`을 **프론티어 밖 known 지뢰만**으로 계산하도록 수정. 초기 250보드 테스트가 즉시 검출 | 불변식 위반은 우회 금지(§4) → 근본수정 | `csp.ts` |
| 2026-06-04 | **[승인] M3 개발 DB = SQLite**(Docker/Postgres 미설치). 포터블 타입(`Uuid`/`JSON`/`Enum(native_enum=False)`)으로 작성해 Postgres 호환 유지. `docker-compose.yml`+Postgres 설정은 CI/운영용으로 함께 작성. `DATABASE_URL`로 dev=sqlite/prod=postgres 전환 | 사용자 승인(블로커 결정) | 기술백서 §3.1(운영은 PG 유지), `.env.example` |
| 2026-06-04 | **서버 Python = 3.14** 고정(venv). `py -0`가 3.12를 보였으나 디스크에 없음(스테일 등록), 실제 3.14만 설치. fastapi/sqlalchemy/alembic/pydantic(-core)/uvicorn 모두 cp314 휠 정상 설치. **주의: M4 `torch`의 3.14 휠 미제공 가능 → M4 진입 시 재확인** | 환경 실측 | 로컬환경 가이드(추후 갱신) |
| 2026-06-04 | 비 ASCII 경로(`내 폴더…`)에서 `py` 런처 venv 생성 실패(ANSI 경로 깨짐). **python.exe 직접 호출**(PowerShell `&`=CreateProcessW)로 해결 | Windows+한글경로 이슈 | — |
| 2026-06-04 | **M4 환경**: uv로 Python **3.12.13** 설치(`uv python install 3.12`; minor-link 경고는 무시 가능, 인터프리터는 정상), `server/.venv` 3.12 재생성. **torch 2.6.0+cu124** 핀(별도 index-url 설치, pyproject 비포함 — 가이드 §2). numpy 2.4.6. pyproject에 `[build-system]`(setuptools) 추가해 `-e .` 가능 | torch 휠/재현성 | `server/pyproject.toml`, 로컬환경 가이드(추후 갱신) |
| 2026-06-04 | **문서 위치 불일치 → 해결**: 5개 백서를 레포 루트에서 `docs/`로 이동(`git mv`). `CLAUDE.md`의 `docs/...` 참조가 모두 유효해짐. README 참조도 `docs/`로 갱신 | 사용자 승인 후 이동(파일트리 §1 트리의 `docs/`와 일치) | `CLAUDE.md`·README |

---

## 세션 로그
> 세션마다 한 줄: 무엇을 했고 어디서 멈췄는지.

- **YYYY-MM-DD**: 프로젝트 문서 세트 정리. 스캐폴딩 미착수. 다음: M1 시작.
- **2026-06-04**: 레포 스캐폴딩(pnpm workspace·tsconfig.base·eslint 경계룰·prettier·.gitignore) + `@msai/core` M1 구현(types/rng/board/difficulty/metrics/env/index). vitest 32/32 그린, tsc·eslint 클린. M1 DoD 5항목 전부 통과. 다음: M2 솔버.
- **2026-06-04**: 문서 5개 `docs/`로 이동(`git mv`). M2 솔버 구현(view/single-point/patterns(subset)/csp/probability/index + 보드 cloneBoard). 거짓양성 버그 1건(바다-경계 이중계상) 검출·근본수정. vitest 40/40, tsc·eslint 클린. M2 DoD 3항목 통과(거짓양성0·NG~100%·확률 일치). 다음: M3 server 골격(환경 확인 필요).
- **2026-06-04**: M3 DB 블로커(Docker/PG 미설치) → 사용자 SQLite 승인. 서버 venv(Python 3.14) + 의존성 설치. FastAPI 골격(config/deps/db/session/models/schemas/api: /health·models CRUD·training·benchmark) + Alembic(초기 마이그레이션, SQLite `upgrade head` 성공, `alembic check` 무드리프트). pytest 3 그린, ruff 클린, uvicorn 실기동 /health 200. M3 DoD 4항목 통과. 다음: **M4 trainer — 단, torch용 Python 3.12 설치 필요(블로커 가능성)**.
- **2026-06-04**: M4 착수. uv로 Python 3.12 설치, venv 재생성, torch 2.6.0+cu124 설치 → `torch.cuda.is_available()` True(RTX 4060). `trainer/env.py`(board.ts 1:1 재현) + `tests/test_env.py` 12 그린 → M4 DoD #4 충족. pytest 15, ruff 클린. **다음: encoding/model/replay/dqn/reward/train 구현 + Beginner 학습→승률 게이트(장시간).**
- **2026-06-04**: M4 트레이너 전부 구현(encoding/model/replay(PER)/dqn(Double DQN·n-step·마스킹)/reward/train/evaluate) + shape 테스트 5. sanity(5×5×3) eval 0.32→0.66 단조상승·loss 안정, 체크포인트 재로드 재현(0.66→0.66) → DoD #2·#3·#4 충족. **Beginner 학습 background 시작(PID `beginner.pid`).** 남은 것: DoD #1(승률 0.85 게이트). pytest 20, ruff 클린.
- **2026-06-04**: 사용자 요청으로 **라이브 터미널 뷰어 `trainer/watch.py`** 추가(rich). AI 플레이(한 수씩·컬러) + 학습 메트릭/승률 스파크라인 실시간. 백그라운드 학습의 체크포인트·로그·CSV를 읽어 반영. smoke 통과, ruff 클린. (M8 웹 대시보드 전까지의 경량 관찰 도구.)
- **2026-06-04**: 사용자 요청으로 **웹 대시보드 `trainer/dashboard.py` + `dashboard.html`** 추가(FastAPI+WebSocket+websockets). 서버가 모델로 플레이→WS로 보드/클릭/풀이시간/승패 스트리밍, 브라우저가 렌더 + `/metrics`로 승률곡선·loss·ε. 다크 계측기 UI(디자인 토큰·anti-cliché 준수). E2E 검증(HTML·/metrics·WS frames), ruff·pytest 그린. Beginner 학습은 계속 진행 중(관찰만).
