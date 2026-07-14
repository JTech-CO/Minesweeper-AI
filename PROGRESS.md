# Minesweeper AI 진행 상태

**갱신일**: 2026-07-15
**현재 phase**: M5 완료
**상태**: COMPLETE - M4 학습 및 M5 metrics/API/DB DoD 통과

## 현재 결론

M4-R2-B constraint-graph policy가 고정 canonical validation에서 beginner 91.1%,
intermediate 65.5%, expert 13.6%를 기록해 direct-transfer 게이트를 통과했다.
해당 checkpoint에서 M4-R3 masked PPO production run을 update 200까지 완료했고, 최종 R3 best는
독립 beginner validation 937/1,000(93.7%)과 bit-identical 재로드를 통과했다.

상세 재현 결과와 시도 내역은 `docs/M4_R2_블로커.md`가 단일 출처다.

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

## 다음 작업

M6 curriculum에서 Beginner→Intermediate→Expert 전이와 가중치 승계를 구현한다.
M4/M5에서 검증된 constraint-graph checkpoint 계보와 canonical seed gate를 그대로 유지한다.

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