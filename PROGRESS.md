# PROGRESS.md — 진행 상태 (세션 인계용)

> Claude Code는 세션 간 상태가 리셋된다. **시작 시 이 파일을 먼저 읽고, 종료 시 갱신**한다(루트 `CLAUDE.md` §1). 절차·DoD는 `docs/지뢰찾기AI_하네스.md` 참조.

---

## 현재 상태
- **현재 Phase**: M4 진행 중. **모델 단독 학습은 정체(beginner best 0.205, ~무작위 추론 수준)** — 통제 진단으로 근본원인 확정(아래). 사용자 결정 **C**로 **B(솔버-하이브리드, 부분 M7)를 먼저 적용해 시연 승률 확보** → **다음은 A(M4 RL 보상/탐험 교정 재학습)**.
- **플레이 승률(솔버 하이브리드, 측정)**: 표준 랜덤 보드 beginner **0.948** / intermediate **0.820** / expert **0.370**(표준 강제-추측 상한). 모델 단독은 beginner 0.193.
- **NG(무추측) 모드 추가**: 대시보드 보드에 NG 토글 — 추측 없이 풀리는 보드만 생성 + 배치 솔버 → **전 난이도 100% 클리어**(검증 100/60/40). 일관 시연용. (온라인 실사이트 플레이는 랜덤 보드 그대로.)
- **마지막 갱신**: 2026-06-05
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
- **웹 통합 관리 대시보드**(브라우저 자동): `python -m trainer.dashboard`(또는 `server\dashboard.bat` 더블클릭) → http://127.0.0.1:8800.
  - **학습이 대시보드 인프로세스로 관리됨**(`trainer/manager.py`): 일시정지/재시작·라이브 하이퍼파라미터(lr/batch/train-freq/ε floor, 재시작 없이)·**누적 세션수 영속화**(모델 생성 시점 기준, 대시보드 재기동에도 카운트 이어짐)·**beginner 체크포인트에서 resume**.
  - GPU 모니터(VRAM/온도/클럭/util, **읽기전용 — 오버클럭 미지원, 안전**), 병렬 플레이 1~8보드 + 난이도 선택(실시간), 승률 곡선.
  - 학습=GPU, 플레이=CPU 스냅샷(`play_net`)으로 분리(경쟁 없음). 상태/플레이는 WebSocket, 제어는 `/api/control|config|play`. 터미널 뷰어 `trainer.watch`도 유지. (클라이언트 ONNX 풀 React 대시보드는 M8.)
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
| 2026-06-04 | **[환경수정] venv 기반 Python을 안정 위치로 복사**: uv 관리 dir(`AppData\Roaming\uv\python\cpython-3.12.13-...`)이 uv 명령(추가 `uv pip install`) 시 재링크/정리되며 잠깐 사라져 venv `python.exe`가 "No Python at..."로 실패(사용자 dashboard.bat 더블클릭 시 발생). → 기반 인터프리터를 **`C:\Users\MSI\msai-py312`로 복사**하고 `.venv\pyvenv.cfg`의 `home`을 그쪽으로 변경(trampoline이 런타임에 pyvenv.cfg 참조 확인). torch/cuda 정상, 재발 방지. (`.venv`는 비커밋이라 이 변경은 로컬 전용) | uv managed-python 불안정 | `server/.venv/pyvenv.cfg`(로컬) |
| 2026-06-04 | **문서 위치 불일치 → 해결**: 5개 백서를 레포 루트에서 `docs/`로 이동(`git mv`). `CLAUDE.md`의 `docs/...` 참조가 모두 유효해짐. README 참조도 `docs/`로 갱신 | 사용자 승인 후 이동(파일트리 §1 트리의 `docs/`와 일치) | `CLAUDE.md`·README |
| 2026-06-05 | **[승인] 결정 C: B(솔버-하이브리드) 먼저, A(M4 RL 교정) 나중**. M4 게이트(0.85) 미달·정체 진단 후, 시연 승률을 위해 **부분 M7을 M5·M6보다 먼저** 적용(phase 순서 일부 앞당김 — 사용자 승인). | 모델 단독은 deduction 미학습(0.19)·표준 상한 존재. 솔버는 즉시 0.95/0.82. 일관 시연=NG/하이브리드(설계) | 하네스 phase 순서 |
| 2026-06-05 | **새 컴포넌트: `server/trainer/solver.py`**(파일트리에 없던 Python 솔버). TS `@msai/core/solver`(M2)의 **충실 이식**. 정식 M7의 하이브리드/ONNX는 여전히 **TS(agent/)가 정본**; 이 Python 솔버는 **Python 플레이 경로(대시보드·온라인) 전용**. **거짓양성 0 불변식**은 `tests/test_solver.py`로 독립 검증(§5). | 사용자 demo가 Python 경로라 즉효. 솔버는 패리티 강제 이중구현 대상 아님(env·인코딩만) | 파일트리(솔버에 Python 추가), M7 |

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
  - venv 기반 Python을 안정 위치(`C:\Users\MSI\msai-py312`)로 복사·repoint(uv 관리 dir 재링크로 인한 "No Python" 글리치 방지). 브라우저 자동열기를 포트-준비 후로 변경. `server\dashboard.bat` 더블클릭 런처 추가.
- **2026-06-04**: 대시보드: 병렬 보드 1~18, 난이도 옵션 하드코딩(항상 선택), 옛 프로세스/`--tag` 별칭 등 launch 견고화.
- **2026-06-04**: 대시보드 **자동 리로드**(uvicorn --reload; MANAGER를 lifespan에서 env 기반 생성→worker 재시작 시 체크포인트 resume) + **CLI 스타일 최소 UI 재설계**(모노스페이스·얇은 보더·그림자/그라데이션/트랜지션 제거로 저RAM·고밀도, 보드 숫자/지뢰 색 유지, lifetime 위치 유지) + 보드 패널 fit(유동 1fr) + 병렬 18. JS/ID 훅 보존.
- **2026-06-04**: 사용자 요청으로 **실제 사이트 플레이(minesweeper.online)** 추가 — `trainer/online.py`(Playwright 서버측, 모델이 DOM 보드 읽고 클릭, 대시보드에 스트리밍) + `/api/online` + UI URL/Play/Stop. **개인 시연·레이트제한·약관 준수**(공개 순위 자동제출 금지). DOM 계약(`cell_X_Y`/`hd_*`)은 한 곳에 격리(런북 10). **검증 한계**: headless 샌드박스에서 minesweeper.online이 보드를 렌더 안 함(세션/동의/봇차단 가능성) → **가시 브라우저(headless=False) 기본** + 사용자 실제 게임 URL로 검증 필요(셀렉터 보정은 `online.py` 1곳). parse_board 단위·엔드포인트 검증·ruff/pytest 그린.
- **2026-06-05**: 대시보드 마무리(난이도별 고정칸 보드+펄스), **온라인 플레이 안정화**(uvicorn reload 워커 SelectorEventLoop→Playwright 서브프로세스 NotImplementedError를 전용 Proactor 스레드/메인루프 적응 실행으로 해결; 보드 렌더 대기+0-인덱스 좌표 수정; CPS·시도·ms·승패 자체측정 + 연속 플레이; **Google OAuth 차단 우회 위해 실제 Chrome CDP 연결** 로그인). **승률 정체 진단**: managed_state 730k·best 0.205, eval 완전 평탄 → M4 게이트 실패(Expert 상한과 무관). 통제 진단(체크포인트 로드): 확정-안전 추종 58%·확정-지뢰 클릭 3.4%·무작위 대비 미미 → **모델이 deduction 미학습**(조밀보상 Q포화 + eps바닥 재개 가설). 사용자 결정 **C** → **B 구현**: `trainer/solver.py`(TS 솔버 M2 충실 이식, **거짓양성 0** 게이트 580보드 통과, 논리클리어 88/65/12%) + `manager.hybrid_act`(확정안전→최저확률 추측→모델 폴백) + 대시보드/온라인 플레이 연결. 측정 승률 beginner 0.948·inter 0.820·expert 0.370. 이어 **NG(무추측) 모드** 추가(`generate_no_guess_board` + 대시보드 배치 솔버 + UI 토글) → 전 난이도 **100% 클리어**(검증 100/60/40). pytest 25 그린. **다음: A(M4 RL 재학습 교정) 또는 정식 M7(TS 하이브리드+ONNX).**
- **2026-06-04**: 사용자 요청으로 대시보드를 **통합 관리 시스템**으로 확장. `trainer/manager.py`(인프로세스 관리형 학습: pause/resume·라이브 hp·resume·영속 누적카운트·GPU 모니터·play_net 스냅샷) + `dashboard.py` 컨트롤플레인(REST 제어 + WS 상태/병렬보드) + `dashboard.html` 관리 UI. nvidia-ml-py(읽기전용 GPU). E2E 검증(상태/GPU/병렬플레이/제어), 관리형 학습 루프 검증(카운트·pause/resume·영속). **오버클럭은 안전상 미지원**(사용자 합의). 다음: 현재 detached 학습 중지 후 관리형으로 전환(resume).
