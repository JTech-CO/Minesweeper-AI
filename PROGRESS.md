# Minesweeper AI 진행 상태

**갱신일**: 2026-07-14
**현재 phase**: M4-R2
**상태**: BLOCKED - expert direct-transfer 게이트 미통과

## 현재 결론

M4-R0 평가/계보와 M4-R1 durable worker는 완료됐다. M4-R2는 기존 CNN과 승인된
R2-A axial 모델까지 구현했으나 고급 policy-only canonical validation이 0/500이어서
게이트를 통과하지 못했다. M4-R3 masked PPO backend와 통합 대시보드는 구현·smoke
검증됐지만, phase 게이트를 우회하지 않기 위해 실제 PPO 학습 run은 시작하지 않았다.

상세 재현 결과와 시도 내역은 `docs/M4_R2_블로커.md`가 단일 출처다.

## Phase 상태

- [x] M1 core 엔진: 결정론, 첫 클릭 안전, cascade, 3BV, vitest 통과.
- [x] M2 core solver: false-positive 0, NG solver clear 게이트 통과.
- [x] M3 server 골격: migration, health, model metadata CRUD 통과.
- [ ] M4 trainer: R0/R1 완료, R2 expert 게이트 blocked, R3 실제 학습 미진입.
- [ ] M5 metrics/API/DB.
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

### M4-R2 표현·교사 사전학습 - blocked

- v2 CNN: beginner 88% / 100, intermediate 26% / 100, expert 0/500.
- 승인된 R2-A:
  - visible-only 20채널 constraint-aware encoding.
  - width 128, residual 8블록, axial row/column attention 2회.
  - expert hard-state/component-balanced teacher data.
  - 난이도별 선택 20,000 상태, 12 epoch.
- R2-A canonical validation:
  - beginner 181/200 = 90.5%.
  - intermediate 68/200 = 34.0%.
  - expert 0/500, 평균 88.778수.
- risk/certainty head 보정 2종도 별도 smoke suite에서 0/32.

### M4-R3 masked PPO - 구현됨, 활성화 금지

- legal-action masked vector rollout.
- GAE, clipped PPO, value/risk auxiliary loss, potential-based progress reward.
- 단조 증가 training seed cursor.
- JSONL metrics, lifetime counter, atomic last/best checkpoint.
- CPU 소형 backend update/checkpoint smoke test 통과.
- R2 expert direct-transfer >0 전까지 production run 시작 금지.

## 로컬 통합 대시보드

실행:

```powershell
cd server
.venv\Scripts\python.exe -m trainer.dashboard
```

또는 `server/dashboard.bat`를 더블클릭한다. 기본 주소는
`http://127.0.0.1:8800`이다.

- durable worker start/pause/resume/stop.
- live learning-rate/entropy 설정.
- GPU utilization/VRAM/temperature read-only 표시.
- persistent train/eval/lifetime curves.
- legacy lifetime 3,253,928회 + 새 PPO lifetime 합산.
- 난이도별 policy-only W/L/win rate/평균 승리 시간 영속 저장.
- 개별 로컬 보드 클릭 수, 클릭 sequence, ms, CPS, 결과 pulse.
- 난이도별 고정 18px cell, 병렬 상한 beginner 8/intermediate 4/expert 2.
- 기존 대시보드는 `dashboard_legacy.py/html`로 보존.
- minesweeper.online 및 실사이트 자동화 없음.

## 다음 작업

STOP 규칙에 따라 사용자 결정 전 추가 모델 실험이나 R3 실제 학습을 하지 않는다.

권장 선택지는 B안 constraint-graph policy다. visible clue와 hidden frontier를 이분
그래프로 만들고 residual mine constraint를 message passing으로 표현한다. solver의
결론을 inference 입력으로 사용하지 않으며, expert direct-transfer >0 게이트는 유지한다.

## 불변식

- 테스트/게이트를 낮추거나 seed/game 수를 바꾸지 않는다.
- solver false-positive-0을 유지한다.
- solver 출력은 policy inference 입력으로 넣지 않는다.
- TS/Python env 및 encoding parity를 깨지 않는다.
- `*.pt`, `*.onnx`, `.env`, runtime state를 커밋하지 않는다.
- 실사이트 자동화와 GPU overclock을 복구하지 않는다.
