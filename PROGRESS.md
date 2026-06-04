# PROGRESS.md — 진행 상태 (세션 인계용)

> Claude Code는 세션 간 상태가 리셋된다. **시작 시 이 파일을 먼저 읽고, 종료 시 갱신**한다(루트 `CLAUDE.md` §1). 절차·DoD는 `docs/지뢰찾기AI_하네스.md` 참조.

---

## 현재 상태
- **현재 Phase**: M1 **완료** → 다음 M2 (core 논리 솔버)
- **마지막 갱신**: 2026-06-04
- **환경 점검**: Node v25.2.0 / pnpm 11.1.3 / git 2.54 확인됨. torch.cuda.is_available() = [ ] (M4) / DB 기동 = [ ] (M3)

---

## Phase 체크리스트
각 phase는 `하네스` §1의 DoD를 **전부** 충족해야 [x].

- [x] **M1** core 엔진 — vitest 그린(32/32)·결정론·첫클릭안전·캐스케이드·3BV
- [ ] **M2** core 솔버 — 거짓양성 0·NG 솔버 ~100%·CSP 확률 일치
- [ ] **M3** server 골격 — alembic head·/health 200·모델 메타 CRUD
- [ ] **M4** trainer — Beginner 승률 게이트 도달·loss 비발산·체크포인트 재현
- [ ] **M5** 메트릭 — WS 수신·체크포인트 DB 기록
- [ ] **M6** 커리큘럼 — 자동 승급·Expert ~40% 수렴
- [ ] **M7** ONNX+agent — 인코딩 패리티·ONNX↔torch 일치·NG 하이브리드 ~100%
- [ ] **M8** web — 실시간 차트·플레이 시각화·토큰/anti-cliché·접근성
- [ ] **M9** adapter — 보드 파싱 일치·클릭 동작·NG 한 판 클리어
- [ ] **M10** 배포 — 벤치 저장/표시·CI 그린·배포 산출물

---

## 다음 할 일 (구체적으로) — M2 core 논리 솔버
1. `packages/core/src/solver/single-point.ts` — 단일점 규칙(숫자=인접플래그→나머지 안전 / 숫자=인접hidden수→전부 지뢰). `Board`의 visible 상태만 사용(mineLayout 비참조).
2. `solver/patterns.ts` — 1-2-1, 1-2-2-1 부분집합 규칙.
3. `solver/csp.ts` — 프론티어 열거(제약 만족 배치) → 확실 안전/확실 지뢰 도출. 큰 프론티어 분할.
4. `solver/probability.ts` — 셀별 지뢰 확률(추측 폴백).
5. `solver/index.ts` — `solveStep` 통합 진입점(단일점→패턴→CSP 순).
6. NG 보드 생성기(reject-sampling): 솔버로 "추측 없이 풀림" 확인하며 생성. `test/solver.test.ts`.
7. **DoD**: 거짓양성 0(대량 무작위 보드) · NG 보드 솔버 단독 ~100% 클리어 · CSP 확률 손계산 케이스 일치. (하네스 M2)

> 주의: 솔버는 플레이어 가시 정보(`revealed`/`adjacent`(공개된 셀만)/`flagged`)만 사용. `mineLayout`을 보면 거짓양성 0이 무의미해짐 → 절대 참조 금지.

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
| 2026-06-04 | **문서 위치 불일치 관찰**: `CLAUDE.md`/하네스는 `docs/...` 경로를 참조하나 실제 5개 문서는 **레포 루트**에 있음. 임의 이동 안 함 | 사용자 결정 영역(불변식: 임의 구조 변경 금지) | 루트 `CLAUDE.md` 문서 인덱스 |

---

## 세션 로그
> 세션마다 한 줄: 무엇을 했고 어디서 멈췄는지.

- **YYYY-MM-DD**: 프로젝트 문서 세트 정리. 스캐폴딩 미착수. 다음: M1 시작.
- **2026-06-04**: 레포 스캐폴딩(pnpm workspace·tsconfig.base·eslint 경계룰·prettier·.gitignore) + `@msai/core` M1 구현(types/rng/board/difficulty/metrics/env/index). vitest 32/32 그린, tsc·eslint 클린. M1 DoD 5항목 전부 통과. 다음: M2 솔버.
