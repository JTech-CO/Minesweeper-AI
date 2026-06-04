# PROGRESS.md — 진행 상태 (세션 인계용)

> Claude Code는 세션 간 상태가 리셋된다. **시작 시 이 파일을 먼저 읽고, 종료 시 갱신**한다(루트 `CLAUDE.md` §1). 절차·DoD는 `docs/지뢰찾기AI_하네스.md` 참조.

---

## 현재 상태
- **현재 Phase**: M2 **완료** → 다음 M3 (server 골격, FastAPI + DB)
- **마지막 갱신**: 2026-06-04
- **환경 점검**: Node v25.2.0 / pnpm 11.1.3 / git 2.54 확인됨. **M3 진입 전 확인 필요**: Docker(`docker compose up -d db`)·Python 3.11+ venv (로컬환경 가이드 §). torch.cuda = [ ] (M4) / DB 기동 = [ ] (M3)

---

## Phase 체크리스트
각 phase는 `하네스` §1의 DoD를 **전부** 충족해야 [x].

- [x] **M1** core 엔진 — vitest 그린(32/32)·결정론·첫클릭안전·캐스케이드·3BV
- [x] **M2** core 솔버 — 거짓양성 0(420+스윕2150보드)·NG 솔버 ~100%(생성10판)·CSP 확률 일치
- [ ] **M3** server 골격 — alembic head·/health 200·모델 메타 CRUD
- [ ] **M4** trainer — Beginner 승률 게이트 도달·loss 비발산·체크포인트 재현
- [ ] **M5** 메트릭 — WS 수신·체크포인트 DB 기록
- [ ] **M6** 커리큘럼 — 자동 승급·Expert ~40% 수렴
- [ ] **M7** ONNX+agent — 인코딩 패리티·ONNX↔torch 일치·NG 하이브리드 ~100%
- [ ] **M8** web — 실시간 차트·플레이 시각화·토큰/anti-cliché·접근성
- [ ] **M9** adapter — 보드 파싱 일치·클릭 동작·NG 한 판 클리어
- [ ] **M10** 배포 — 벤치 저장/표시·CI 그린·배포 산출물

---

## 다음 할 일 (구체적으로) — M3 server 골격 (FastAPI + DB)
> **진입 전 환경 확인**: `docker compose up -d db`(Postgres16) 가능 여부, Python 3.11+ venv. 안 되면 멈춤 규칙(§4)으로 보고.
1. `server/pyproject.toml` 의존성(fastapi·uvicorn·sqlalchemy·alembic·pydantic·psycopg). venv 생성.
2. 루트 `docker-compose.yml`(db: postgres16) + `.env.example`(DATABASE_URL·CORS_ORIGINS·MODEL_STORAGE_DIR).
3. `server/app/core/config.py`(.env 로드) · `db/session.py` · `db/models.py`(TrainingRun/Model/BenchmarkResult, 기술백서 §2.3).
4. Alembic init + 최초 마이그레이션(env.py에 ORM import — 런북 7). `alembic upgrade head`로 테이블 생성.
5. `schemas/`(pydantic) — TS `@msai/core` 타입과 계약 일치(수동 점검).
6. `app/main.py` — `GET /health` 200 + 모델 메타 list/get/put(트레이너 의존부 스텁 가능). `server/tests/`(pytest).
7. **DoD**: alembic head 성공 · uvicorn 기동 `/health` 200 · 모델 메타 CRUD 왕복 · pydantic↔TS 타입 무모순. (하네스 M3)

> 참고: M3는 Python 영역. 게임 규칙/인코딩 이중 구현은 M4부터(`env.py`는 M1 `board.ts` 규칙을 1:1 재현 — 특히 첫클릭 `safe-area`, flat row-major, 캐스케이드).

---

## 미결 질문 / 블로커 (사용자 결정 필요)
> 멈춤 규칙(하네스 §3)으로 멈췄을 때 여기에 적는다: 증상 / 재현 방법 / 시도한 것 / 가설 / 필요한 결정.

- (없음)

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
| 2026-06-04 | **문서 위치 불일치 → 해결**: 5개 백서를 레포 루트에서 `docs/`로 이동(`git mv`). `CLAUDE.md`의 `docs/...` 참조가 모두 유효해짐. README 참조도 `docs/`로 갱신 | 사용자 승인 후 이동(파일트리 §1 트리의 `docs/`와 일치) | `CLAUDE.md`·README |

---

## 세션 로그
> 세션마다 한 줄: 무엇을 했고 어디서 멈췄는지.

- **YYYY-MM-DD**: 프로젝트 문서 세트 정리. 스캐폴딩 미착수. 다음: M1 시작.
- **2026-06-04**: 레포 스캐폴딩(pnpm workspace·tsconfig.base·eslint 경계룰·prettier·.gitignore) + `@msai/core` M1 구현(types/rng/board/difficulty/metrics/env/index). vitest 32/32 그린, tsc·eslint 클린. M1 DoD 5항목 전부 통과. 다음: M2 솔버.
- **2026-06-04**: 문서 5개 `docs/`로 이동(`git mv`). M2 솔버 구현(view/single-point/patterns(subset)/csp/probability/index + 보드 cloneBoard). 거짓양성 버그 1건(바다-경계 이중계상) 검출·근본수정. vitest 40/40, tsc·eslint 클린. M2 DoD 3항목 통과(거짓양성0·NG~100%·확률 일치). 다음: M3 server 골격(환경 확인 필요).
