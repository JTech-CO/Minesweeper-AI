# 지뢰찾기 AI 재구성 기획

**작성일**: 2026-07-13  
**상태**: 구현 전 제안  
**범위**: 로컬 전용 학습·평가·대시보드. 실사이트 자동화는 복구하지 않는다.

## 1. 결론

현재 병목은 학습량이 아니라 **목적함수, 평가, 표현, 실행 구조**다. 기존 가중치를
그대로 더 학습하거나 `width=128`, `blocks=6`, Dueling만 추가하는 방식은 우선순위가
아니다. 현재 모델은 3.25M episode를 학습했지만 독립 시드 평가에서 초급 22.6%,
중급·고급 0%였고, 마지막 가중치는 초급 17.8%로 퇴행했다.

권장 방향은 다음과 같다.

1. 평가·체크포인트·실험 계보를 먼저 신뢰할 수 있게 만든다.
2. 대시보드 프로세스와 학습 워커를 분리하고 학습 루프를 하나로 합친다.
3. 논리 솔버를 **교사 데이터 생성기**로 사용해 공간 정책을 사전학습한다.
4. 전역 문맥을 갖는 policy/value/risk 멀티헤드 모델을 masked PPO로 미세조정한다.
5. 초급 M4 게이트를 통과한 뒤에만 M5와 M6 커리큘럼으로 진행한다.

## 2. 감사 기준선

- TypeScript: vitest 40/40, typecheck 통과.
- Python: pytest 25/25, ruff 통과.
- 저장 상태: 3,253,928 episodes, 25,740,972 steps, 기록상 best eval 0.35.
- 최근 100개 eval 표본 평균: eval 0.2134, train 0.1496.
- 실행 중인 학습 프로세스: 없음.

기존 평가셋과 겹치지 않는 `seed_base=1_700_000_000`에서 평가했다.

| 체크포인트        |          초급 |       중급 |       고급 |
| ----------------- | ------------: | ---------: | ---------: |
| `managed_best.pt` | 22.6% / 1,000 | 0.0% / 300 | 0.0% / 200 |
| `managed_last.pt` | 17.8% / 1,000 | 0.0% / 300 | 0.0% / 200 |

기존 고정 200판에서는 best가 정확히 34.5%였다. 대시보드의 35%는 일반화 승률이
아니라 같은 200판을 반복 평가하면서 가장 높은 표본을 선택한 값에 가깝다.

`managed_state.json`은 episode 2,786,000의 eval을 23.5%로 기록하지만
`managed_best.pt`는 같은 episode와 34.5%를 메타데이터로 갖는다. 현재 파일명만으로는
재현 가능한 run을 식별할 수 없다.

## 3. 핵심 문제

### P0. 평가 과대추정

`evaluate.py`는 매번 같은 200개 시드를 사용하고, `manager.py`는 2,000 episode마다
그 표본의 최고값을 저장한다. train/validation/test 시드 분리, 신뢰구간, 대규모 최종
평가가 필요하다.

### P0. 보상 목적 불일치

합법 행동 마스킹 상태에서 안전 셀은 거의 항상 `+0.3`, 지뢰는 `-1`, 승리는
`+1`이다. 한 칸과 대규모 캐스케이드가 같은 보상을 받고 step cost도 0이므로 빠른
클리어가 아니라 다음 클릭의 생존과 누적 양의 보상을 최적화한다.

### P0. 표현력 부족

현재 QNet은 약 30만 파라미터의 width 64, residual block 4개, 단일 Q head다.
백서의 Dueling은 구현되지 않았다. 수용영역은 약 19x19라 30열 고급 보드의 전역 제약을
볼 수 없고 입력에도 frontier, 진행률, 유효 영역, 전역 숨은 셀 수가 없다.

### P0. 실행·저장 구조 불안정

학습은 자동 리로드가 기본인 대시보드의 daemon thread에서 실행된다. 리로드 때 replay는
비워지고 RNG·라이브 설정은 저장되지 않으며 종료 join이 없다. 상태 JSON은 atomic
replace 없이 쓰고 복구 오류는 넓은 `except`로 숨긴다.

`train.py`와 `manager.py`도 서로 다른 capacity, warmup, target sync, beta,
LR schedule을 구현한다. 정식 `app.main` API는 읽기 전용 M3 스켈레톤이고 실제 제어는
별도 `trainer.dashboard`에 있어 동일 실험을 재현하기 어렵다.

### P1. 커리큘럼·회귀 게이트 부재

계획된 `curriculum.py`가 없고 manager는 한 난이도만 학습한다. 테스트는 shape,
finite loss, 규칙, 솔버 건전성만 검사하며 장기 학습 추세, resume, checkpoint 계보,
평가 분리, Python/TS encoding parity를 검증하지 않는다.

### P1. 모델 지표와 제품 지표 혼합

대시보드 플레이는 solver-first이고 모델은 마지막 fallback이다. 초급 94.8%, 중급
82.0%, 고급 37.0%는 제품 지표지만 RL 모델 승률과 분리해야 한다. 이후 모든 리포트는
`policy-only`, `solver-only`, `hybrid`를 별도 series로 저장한다.

## 4. 목표 구조

```text
server/
  app/
    main.py                 # 유일한 FastAPI 엔트리
    api/                    # run/control/checkpoint/benchmark
    ws/metrics.py           # worker 이벤트 중계
  trainer/
    config.py               # 단일 TrainConfig + config hash
    experiment.py           # 유일한 학습 orchestration
    env.py
    encoding.py             # v2 채널 계약
    models/policy_value.py  # policy/value/risk/certainty heads
    algorithms/ppo.py       # 주 학습기
    algorithms/dqn_legacy.py
    data/teacher.py         # 솔버 기반 라벨 생성
    evaluation/suites.py    # train/val/test seed registry
    evaluation/metrics.py   # Wilson CI, 3BV, calibration
    storage/checkpoint.py   # atomic save + manifest
    worker.py               # 대시보드와 분리된 프로세스
```

대시보드는 worker를 직접 소유하지 않는다. `app.main`이 run 상태를 DB에서 읽고 별도
worker 프로세스에 제어 명령을 전달한다. 개발 서버 리로드가 발생해도 학습은 계속된다.

## 5. 모델·학습 재설계

### 입력·모델 v2

- 기존 hidden + revealed 0..8 one-hot을 유지한다.
- mine density, hidden ratio, reveal progress, frontier, valid-cell mask를 추가한다.
- 첫 클릭은 중앙 고정 정책으로 분리해 보드 생성과 정책 평가의 결합을 제거한다.
- width 96~128 dilated residual blocks와 global context broadcast를 사용한다.
- masked policy, state value, cell mine-risk, certainty head를 함께 학습한다.
- BatchNorm은 계속 제외하고 GroupNorm과 normalization-free residual을 비교한다.

### 교사 사전학습

검증된 false-positive-0 솔버를 학습 데이터 생성기로 활용한다.

- 표준/NG, 3난이도 상태를 solver trajectory와 hard-state sampling으로 수집한다.
- 실제 mine map, CSP 확률, certain-safe/mine, solver 추천 행동을 라벨로 저장한다.
- Brier/NLL, certain-safe precision, policy action safety를 검증한다.
- 사전학습 뒤 순수 policy-only 승률을 측정해 RL 시작 체크포인트로 고정한다.

### 강화학습

주 알고리즘은 **masked PPO actor-critic**을 권장한다. 공간 policy와 자연스럽게 결합되고
벡터 환경 rollout을 사용하며 replay 복구 문제를 제거한다. 기존 Double DQN은 동일
benchmark의 legacy baseline으로 남긴다.

- 기본 목적: win `+1`, loss `-1`.
- 중간 보상: `safe_revealed / safe_cells`의 potential-based delta.
- 속도 보상은 승률 게이트 이후 작은 3BV-normalized step cost로 별도 fine-tune한다.
- 약한 imitation/risk auxiliary loss를 유지해 deduction 망각을 막는다.

## 6. 구현 순서와 게이트

모든 작업은 현재 M4 안에서 진행하며 M4를 우회해 M5로 표시하지 않는다.

### M4-R0. 평가·계보 복구

- 현재 best/last를 read-only legacy artifact로 동결한다.
- train/validation/test seed registry와 95% Wilson CI를 도입한다.
- checkpoint manifest에 run UUID, git SHA, config/arch hash, current/best eval,
  optimizer/scheduler/RNG 상태를 기록하고 atomic save한다.
- 게이트: 같은 checkpoint·suite 결과가 완전 재현되고 metrics와 manifest가 일치.

### M4-R1. 실행 구조 통합

- `train.py`와 `manager.py`를 하나의 `ExperimentRunner`로 합친다.
- dashboard thread 학습을 별도 worker process로 전환한다.
- config 검증, graceful stop/join, resume 통합 테스트를 추가한다.
- 게이트: 대시보드 재시작 중 episode·optimizer·scheduler 상태 보존.

### M4-R2. 표현·교사 사전학습

- encoding v2 parity fixture, teacher dataset, model v2 및 ablation을 작성한다.
- 게이트: false-positive-0 유지, 세 난이도 direct-transfer가 0%에서 벗어나고
  policy-only 초급이 legacy 22.6%를 통계적으로 상회.

### M4-R3. RL 미세조정

- 사전학습 checkpoint에서 vectorized masked PPO를 시작한다.
- 모델 선택은 validation lower confidence bound, 최종 보고는 test suite 한 번만 사용.
- **M4 DoD를 낮추지 않는다**: 초급 독립 평가 0.85, loss 비발산, checkpoint 재현을
  모두 통과해야 M4 완료.

### 이후

- M5: ExperimentRun/Checkpoint/Benchmark를 정식 API·DB·WS에 연결한다.
- M6: 초급→중급→고급 curriculum과 이전 난이도 rehearsal을 도입한다.
- 난이도 승률은 합산하지 않는다. 고급 표준은 문서상 약 40% 상한, NG는 일관 시연용이다.
- M7~M10은 로컬 전용 범위로 기술백서·파일트리·하네스를 개정한 뒤 진행한다.

## 7. 보존·교체 범위

**보존**

- `packages/core` 엔진·솔버와 거짓양성 0 테스트.
- Python 환경 규칙과 env parity 테스트.
- 로컬 대시보드의 보드 시각화·GPU 모니터·누적 통계 UX.
- 기존 checkpoint는 `legacy-v1` 기준선으로 read-only 보존.

**교체 또는 통합**

- 현재 reward preset, QNet 단일 head, 고정 200판 evaluator.
- `train.py`/`manager.py` 이중 학습 루프.
- dashboard daemon thread 학습과 auto-reload 결합.
- 파일명 기반 checkpoint 계보와 비원자 상태 JSON.
- 정식 API와 별도로 존재하는 dashboard control plane.

## 8. 첫 구현 단위

승인 후 첫 변경은 모델이 아니라 **M4-R0 평가·계보 복구**다. 이 단계가 끝나기 전에는
새 reward나 모델의 성능을 판단할 수 없다. 이후 M4-R1로 실행 구조를 통합하고,
M4-R2부터 새 모델과 교사 학습을 시작한다.
