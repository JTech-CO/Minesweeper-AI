# Minesweeper AI 진행 상태

**갱신일**: 2026-07-16
**현재 phase**: M6 진행 중
**상태**: BLOCKED - M6 Expert 40% 게이트, R4와 유효 R5 방법 실패 후 STOP

## 현재 결론

M6-R4 constraint-posterior-v5 teacher checkpoint가 고정 canonical validation에서
beginner 944/1,000(94.4%), intermediate 145/200(72.5%),
expert 137/500(27.4%, 95% Wilson 23.67-31.47%)를 기록했다.
Expert CI 하한이 이전 R2-B 13.6%를 상회했고 모든 출력 head의 bit-identical 재로드를 통과했다.

두 번째 R4-2 production은 action temperature 0.25, teacher KL 0.10,
learning rate 5e-6, 분리 mine auxiliary 0.01, AMP, 25% rehearsal을 적용했다.
Beginner는 update 18에서 rolling 500게임 94.8%로 승급했고 Intermediate는
update 116에서 rolling 500게임 75.0%로 Expert에 승급했다. 두 번째 R4-2는
update 600 rolling 25.6%, smoke 8/32(25.0%)로 정체되어 update 604에서 중단했다.
R4-3 DAgger도 최종 Expert smoke 7/32(21.875%)로 실패했다.

M4-R2-B와 기존 M6 실패 계보는 docs/M4_R2_블로커.md,
docs/M6_Expert_블로커.md가 단일 출처다.
## Phase 상태

- [x] M1 core 엔진: 결정론, 첫 클릭 안전, cascade, 3BV, vitest 통과.
- [x] M2 core solver: false-positive 0, NG solver clear 게이트 통과.
- [x] M3 server 골격: migration, health, model metadata CRUD 통과.
- [x] M4 trainer: R0/R1/R2/R3 완료, 50 tests 및 최종 DoD 통과.
- [x] M5 metrics/API/DB: WS stream, checkpoint file, Model DB row 게이트 통과.
- [ ] M6 curriculum.
- [ ] M7 ONNX/agent parity.
- [ ] M8 web product dashboard.
- [ ] M9 adapter: 로컬 전용 결정으로 실사이트 자동화 범위에서 제외 예정.
- [ ] M10 배포/CI/benchmark.

## M4 재구성 상태

### M4-R0 평가·계보 - 완료

- immutable smoke/validation/test seed suite.
- 95% Wilson interval.
- legacy best/last read-only 동결.
- atomic v2 checkpoint: run/checkpoint UUID, config/architecture hash, SHA256,
  optimizer/scheduler/RNG, manifest.
- canonical fixed-center baseline: legacy best beginner 28.5% / 1,000.

### M4-R1 실행 구조 - 완료

- 단일 `ExperimentConfig`와 config hash.
- algorithm-independent `ExperimentRunner`.
- 대시보드와 분리된 detached worker process.
- pause/resume/stop, 상태 파일, process 재접속 lifecycle 테스트.

### M4-R2 표현·교사 사전학습 - 완료

- R2-A axial 실패 계보는 docs/M4_R2_블로커.md에 보존.
- R2-B: visible clue↔hidden cell recurrent constraint-graph 4회 + 기존 axial context.
- 세 난이도를 16x30 valid-mask로 padding해 난이도별 10,000 상태를 mixed batch 학습.
- 8 epoch teacher loss 3.0923→2.9113, NaN/발산 없음.
- canonical validation:
  - beginner 911/1,000 = 91.1%.
  - intermediate 131/200 = 65.5%.
  - expert 68/500 = 13.6%, 평균 131.948수.
- solver 출력은 teacher label에만 사용하고 policy inference 입력에는 사용하지 않음.

### M4-R3 masked PPO - 완료

- legal-action masked vector rollout, GAE, clipped PPO, value/risk auxiliary loss.
- canonical graph checkpoint에서 production run storage/runs/m4-r3-graph 시작.
- update 200: 누적 6,510세션, train win 92.0%, smoke 31/32, loss 0.0541.
- R3 final best independent beginner validation 937/1,000 = 93.7%, 평균 16.682수.
- policy/risk/certainty/value tensor의 bit-identical checkpoint 재로드 통과.
- Windows checkpoint/JSON reader 공유 충돌 bounded retry와 CUDA RNG resume 수정.

## 로컬 통합 대시보드

실행:

```powershell
cd server
.venv\Scripts\python.exe -m trainer.dashboard
```

또는 `server/dashboard.bat`를 더블클릭한다. 기본 주소는
`http://127.0.0.1:8800`이다.

- canonical m4-r3-graph worker start/pause/resume/stop.
- live learning-rate/entropy 설정.
- GPU utilization/VRAM/temperature read-only 표시.
- persistent train/eval/lifetime curves.
- legacy lifetime 3,253,928회 + 새 PPO lifetime 합산.
- 난이도별 policy-only W/L/win rate/평균 승리 시간 영속 저장.
- 개별 로컬 보드 클릭 수, 클릭 sequence, ms, CPS, 결과 pulse.
- 난이도별 고정 18px cell, 병렬 상한 beginner 8/intermediate 4/expert 2.
- 기존 대시보드는 `dashboard_legacy.py/html`로 보존.
- minesweeper.online 및 실사이트 자동화 없음.

## M5 메트릭 스트림 + 체크포인트 영속화 - 완료

- trainer/events.py: versioned events.jsonl을 fsync append하고 mutable checkpoint를 고유 spool로 snapshot.
- app/ws/metrics.py: 100ms batch/throttle, 최근 metric replay, bounded client queue.
- WS payload: episode, train/eval win rate, loss, steps, algorithm, difficulty, exploration 값.
- PPO는 epsilon-greedy가 아니므로 epsilon 필드는 null, entropy_coef는 별도 전송.
- app/storage/weights.py: SHA-256 검증, path traversal 차단, content-addressed 저장.
- 같은 볼륨에서는 hardlink를 사용하고 DB commit 이후 spool을 정리해 물리적 가중치 중복 방지.
- TrainingRun/Model row는 deterministic UUID로 idempotent upsert.
- canonical update-200 실제 게이트:
  - ws://127.0.0.1:8000/ws/metrics에서 update 200, episode 6,510, train 92.0%, smoke eval 96.875% 수신.
  - Model c0f9352a-b611-4b6e-8c7a-3a10d86c7507, canonical validation 93.7% 기록.
  - 저장 SHA-256 d5afc601a3127229ab3864398fcfc448504b233a6daa1fb8d89ba7d859155129 일치.
  - 저장 checkpoint link_count=2로 validated 원본과 물리적 데이터 공유.
  - server pytest 54 passed, ruff green.

M5 API 실행:

```powershell
cd server
$env:TRAINER_RUN_DIR = "storage/runs/m4-r3-graph"
$env:MODEL_STORAGE_DIR = "storage/models"
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

## M6-R4 constraint-posterior-v5 - 진행 중

- R4-0:
  - visible-only count-aware factor graph 8회와 posterior/mine/certainty/policy/value 분리 head.
  - solver inference 비호출, hidden mine-layout 독립, 9x9→16x30 padding 불변, illegal mask.
  - padded encode_v2의 비연속 reshape 복사 결함 수정.
- R4-1:
  - production teacher states: beginner 6,000 / intermediate 10,000 / expert 20,000,
    Expert 3배 가중, in-memory only.
  - held-out Expert posterior: Brier 0.002723, NLL 0.403235,
    certain-safe precision 99.51%.
  - canonical validation: beginner 94.4%, intermediate 72.5%, expert 27.4%.
  - checkpoint e02d1109425443e688a89ddc919fcdc1, 모든 출력 bit-identical.
- R4-2:
  - frozen anchor checkpoint는 위 teacher checkpoint로 고정.
  - 첫 시도는 높은 behavior entropy로 Expert 성능이 teacher보다 악화되어 update 204에서 정상 중단.
  - 두 번째 시도는 action temperature 0.25, teacher KL 0.10,
    mine auxiliary 0.01, learning rate 5e-6, base entropy 0.0.
  - Beginner update 18에서 500게임 94.8%로 Intermediate 승급.
  - update 116에서 Intermediate 500게임 75.0%로 Expert 승급.
  - Expert update 150 smoke 7/32(21.875%): 첫 시도의 3/32보다 개선.
  - CUDA+AMP, BLAS/OpenMP thread 1개, dashboard no-reload, live board 1개로 실행.
  - update 600 rolling 500게임 25.6%, smoke 8/32(25.0%).
  - update 604 rolling 25.8%에서 정상 중단. NaN/OOM 없음.
- R4-3:
  - exact solver-only Expert smoke 17/32(53.125%)로 teacher ceiling 확인.
  - policy 방문 상태 12,000개씩 2 rounds를 solver로 재라벨링한 DAgger 구현.
  - round 1 Expert 6/32(18.75%), round 2 Expert 7/32(21.875%).
  - final posterior Brier 0.002164, NLL 0.414180, certain-safe precision 98.76%.
  - Beginner 30/32, Intermediate 24/32는 유지했지만 Expert 40% 미달.
- 검증: server pytest 77 passed, Ruff green, core vitest 40 passed,
  TypeScript typecheck 및 workspace lint green.
## 다음 작업

서로 다른 R4 방법 3개가 실패했으므로 STOP한다. 새로운 M6-R5 아키텍처 또는
제품 목표/게이트 변경은 사용자 명시 승인 전에는 진행하지 않는다. M6를 완료
처리하거나 M7로 진행하지 않는다.

## 불변식

- 테스트/게이트를 낮추거나 seed/game 수를 바꾸지 않는다.
- solver false-positive-0을 유지한다.
- solver 출력은 policy inference 입력으로 넣지 않는다.
- TS/Python env 및 encoding parity를 깨지 않는다.
- `*.pt`, `*.onnx`, `.env`, runtime state를 커밋하지 않는다.
- 실사이트 자동화와 GPU overclock을 복구하지 않는다.

## 2026-07-15 런타임 저장공간 유지보수

- 대시보드 WebSocket 종료 후 완료된 board task 예외가 회수되지 않아 task가 재생성되고 `dashboard.err`가 무제한 증가하던 원인을 수정했다.
- 연결 종료 시 자식 task 결과를 감독하고, 모든 status/board task를 cancel+gather하여 예외를 회수한다.
- 앱 프로세스 stderr를 `dashboard-runtime.err`로 분리하고 파일당 10 MiB, 백업 2개(총 약 30 MiB)로 회전한다.
- 연결 종료 회귀 테스트와 회전 로그 상한 테스트를 추가했다.
- 서버 전체 pytest 56 passed, ruff green.
- 실제 정리에서 대형 `dashboard.err` 7.559 GiB, 재생성 캐시 67.52 MiB, 동일 체크포인트 물리 중복 43.22 MiB를 제거했다.
- C: 실제 여유 공간은 7.656 GiB 증가했고 프로젝트 논리 용량은 약 3.97 GiB로 감소했다.
- validated/best/model SHA-256 `d5afc601a312...` 일치를 재확인했고 hardlink 3개가 같은 데이터를 공유한다.
- 재시작 후 dashboard 8800과 M5 API 8000이 HTTP 200이며, 실제 WebSocket 종료 뒤 3초간 오류 로그 증가량은 0 byte였다.

## M6 커리큘럼 학습 - 진행 중

- `CurriculumController`가 최근 500게임 창과 3회 연속 확인으로 0.85/0.60/0.40 게이트를 관리한다.
- 승급 시 모델 tensor는 bit-identical로 유지하고 환경, optimizer, entropy만 새 단계에 맞게 재설정한다.
- 체크포인트는 `bootstrap.pt`, `last.pt`, `best.pt` 세 경로만 사용하며 routine 저장은 덮어쓴다. M5 저장소 발행은 curriculum 완료 시 1회로 제한했다.
- server 전체 pytest 63 passed, ruff green. bootstrap 대시보드 집중 테스트 4 passed.
- production run: `server/storage/runs/m6-curriculum`, CUDA RTX 4060, 총 5,000 update 상한.
- Beginner→Intermediate: update 18, episode 578, 최근 500게임 91.4%.
- Intermediate→Expert: update 91, episode 1,101, 최근 500게임 60.8%.
- Expert 첫 smoke: update 100, 3/32 = 9.38%. update 108 기준 학습 계속 실행 중.
- run 파일은 21개, 체크포인트 논리 용량 101.93 MiB, checkpoint spool 0개, 오류 로그 0 byte.
- 남은 DoD: Expert 학습 곡선이 약 40% 근방으로 수렴하고 500게임/3회 확인 게이트를 통과해야 한다.

## 2026-07-15 대시보드 플레이 컨트롤 수정

- WebSocket 상태 갱신이 1초마다 사용자가 선택 중인 난이도와 병렬 보드 수를 서버의 이전 값으로 덮어쓰던 결함을 수정했다.
- 플레이 컨트롤에 dirty/pending 상태를 추가해 선택 중인 값은 보존하고, /api/play 적용 성공 후 서버 응답과 동기화한다.
- 난이도별 보드 상한은 beginner 8, intermediate 4, expert 2를 유지한다.
- 실행 중인 8800 대시보드에서 intermediate 3개(16x16), expert 2개(16x30), beginner 6개(9x9) WebSocket 생성을 검증하고 beginner 4개로 복원했다.
- 대시보드 집중 테스트 7 passed, 관련 Ruff 검사 green.
- M6 worker는 영향 없이 Expert update 159, episode 1,431에서 계속 실행 중이다.
## 2026-07-15 M6 Expert STOP

- production curriculum은 update 4,009에서 정상 종료했다. Expert 최근 500게임 최고 9.6%, smoke 최고 15.625%, 종료 smoke 9.375%로 40% 게이트를 통과하지 못했다.
- risk 결합, ground-truth safety 보조손실, 새 seed mixed solver-teacher 추가학습을 서로 다른 방법으로 검증했지만 모두 Expert 수렴을 해결하지 못했다.
- 설계에만 있던 이전 난이도 rehearsal을 mixed-size padded rollout으로 구현했고 현재 난이도 outcome만 게이트에 집계하도록 테스트했다.
- 실패한 임시 가중치 67.96 MiB와 33.98 MiB를 각각 제거했으며 기존 M6 best/last와 metrics는 보존했다.
- 상세 증상, 시도, 결과, 가설은 docs/M6_Expert_블로커.md가 단일 출처다.
- 하네스 STOP 규칙에 따라 M6-R4 아키텍처 변경 승인 전에는 M7로 진행하지 않는다.

## 2026-07-16 M6-R5 STOP

- 사용자 승인으로 constraint-ranker-v6, v5 완전 전이, ranking-aware loss,
  tail-error metrics, on-policy CUDA distillation 경로를 구현했다.
- 계약: hidden mine independence, inference solver 금지, mixed-size padding,
  strict class priority, bounded preference, bit-identical reload.
- certainty-only Expert 기준은 55/200 = 27.5%였다.
- soft combined rank는 소규모 62/200 = 31.0%, 중간 규모 61/200 = 30.5%.
- certainty/guess 분리 rank는 smoke 10/32로 tail 지표를 개선하지 못했다.
- strict class rank의 최초 수치는 legal-mask 결함으로 무효화했다.
- 수정된 유효 checkpoint는 smoke 8/32, canonical 44/200 = 22.0%
  (Wilson 95% CI 16.82-28.24%)였다.
- R5-1 진입 게이트 35%와 M6 최종 40% 게이트에 실패했다.
- legal-mask 수정 strict까지 35%에 실패해 STOP을 확정했다. production curriculum과 M7은 시작하지 않는다.
- 실패 runtime checkpoint는 배포하지 않고 정리한다. 세부 근거는
  docs/M6_R5_설계.md와 docs/M6_Expert_블로커.md가 단일 출처다.

## 2026-07-17 A 구성 안전 경량화

- 모델·checkpoint·optimizer·RNG·metrics·DB는 변경하지 않았다.
- __pycache__/.pyc와 tool cache 1,835개(39,639,535 bytes)를 삭제했다.
- 비활성 .out/.err/PID 52개(124,721 bytes)를 삭제하고 실행 중인
  m6-r4 대시보드 runtime 파일 5개는 보존했다.
- SHA-256이 동일한 venv 파일 2그룹을 NTFS hardlink로 통합해 물리 공간
  6,140,587 bytes를 추가 절감했다.
- 논리 용량은 4,531,235,329 bytes에서 4,491,471,073 bytes로 감소했고,
  계산상 총 물리 절감량은 45,904,843 bytes다.
- PyTorch 2.6.0+cu124, CUDA 12.4, RTX 4060 인식, dashboard HTTP 200,
  server pytest 86 passed를 확인했다.
- m6-r4/bootstrap.pt와 M4 validated checkpoint SHA-256은 정리 전후 동일하다.
