# M6 Expert 수렴 블로커

**갱신일**: 2026-07-16
**상태**: BLOCKED - R4-2와 R4-3 실패, 세 방법 이후 STOP

## 증상

Beginner와 Intermediate 자동 승급은 각각 91.4%, 60.8%로 통과했지만
Expert policy-only 승률이 40%에 수렴하지 않았다. production run은
storage/runs/m6-curriculum update 4,009에서 원자적 last.pt를 저장하고
정상 종료했다.

- Expert 최근 500게임 최고 승률: 9.6%.
- Expert smoke 최고 승률: 5/32 = 15.625%.
- 종료 checkpoint smoke: 3/32 = 9.375%.
- loss는 NaN이나 발산 없이 약 0.02-0.07 범위였다.

## 시도와 결과

| 시도 | 고정 평가 결과 | 결론 |
| --- | ---: | --- |
| 기존 curriculum PPO 장기 학습 | 최고 smoke 5/32 | 약 3,900 Expert update 뒤에도 정체 |
| risk head 결합 0, 0.5, 1, 2, 4 | 모두 3/32 | risk 점수가 행동 순위를 개선하지 못함 |
| ground-truth safe 균등 보조손실 30 update | 6/32 -> 0/32 | 논리적으로 유용한 다음 수 순위를 파괴해 폐기 |
| 새 seed mixed solver-teacher 30,000 states, 8 epoch | beginner 29/32, intermediate 19/32, expert 1/32 | 쉬운 난이도는 보존했지만 Expert 일반화 악화 |

실패한 임시 checkpoint와 로그는 제거했다. 기존 bootstrap.pt, best.pt,
last.pt, metrics와 curriculum 전이 기록은 보존했다.

## 확인된 구현 결함과 수정

설계 문서에 있던 이전 난이도 rehearsal이 실제 PPO rollout에는 없었다.
Intermediate와 Expert에서 25%의 환경을 이전 난이도로 구성하도록 구현했다.
서로 다른 크기는 현재 단계 보드 크기로 zero-padding하고 valid/legal mask로
차단한다. 승급 window에는 현재 난이도에서 종료된 게임만 집계한다.

집중 테스트 8개와 Ruff는 통과했지만, 이 수정만으로 40% 성능 게이트가
입증된 것은 아니다.

## M6-R4 실행 계보

R4-1 constraint-posterior-v5 teacher checkpoint는 canonical validation에서
Beginner 94.4%, Intermediate 72.5%, Expert 27.4%를 기록했다.
Expert Wilson 95% CI 23.67-31.47%로 기존 R2-B 13.6%를 통계적으로 상회했다.

R4-2 첫 production 시도는 learning rate 2e-5, teacher KL 0.02,
mine auxiliary 0.05, base entropy 0.003, action temperature 1.0을 사용했다.

- Beginner: update 17, rolling 500게임 89.2%로 승급.
- Intermediate: update 156, rolling 500게임 60.2%로 승급.
- Expert: update 200 smoke 3/32 = 9.375%, rolling 2.33%.
- teacher checkpoint smoke 9/32 = 28.125%보다 크게 악화되어 update 204에서 정상 중단.
- NaN/OOM은 없었고 실패 원인은 높은 behavior entropy와 약한 anchor 보존으로 판단했다.

실패 가중치는 발행하지 않는다. 두 번째 시도는 action temperature 0.25,
teacher KL 0.10, learning rate 5e-6, mine auxiliary 0.01,
base entropy 0.0과 restart entropy 0.001로 teacher 정책 보존과
near-greedy rollout을 강화한다. 40% 게이트와 seed/game 수는 변경하지 않는다.

두 번째 시도는 Beginner를 update 18, rolling 500게임 94.8%로 통과했다.
컴퓨터 업데이트 전 update 43, Intermediate stage update 25,
현재 창 승률 75.19%에서 정상 중단했다. NaN/OOM/worker 오류는 없고
storage/runs/m6-r4/last.pt와 manifest를 보존해 동일 run에서 재개했다.

재개 시 CUDA+AMP를 유지하고 BLAS/OpenMP thread를 1개로 제한했으며,
dashboard auto-reload를 끄고 live board를 1개로 줄였다. worker CPU는 실측
0.93 core, RAM 1.98GB, Expert VRAM은 약 5.5GB였다.

- Intermediate: update 116, rolling 500게임 75.0%로 Expert 승급.
- Expert update 150 smoke: 7/32 = 21.875%.
- 첫 시도 update 200 smoke 3/32보다 개선됐지만 40% 게이트는 아직 미달이다.
## M6-R4-2 최종 판정

- update 500: rolling 20.8%, smoke 7/32 = 21.875%.
- update 550: rolling 23.0%, smoke 5/32 = 15.625%.
- update 600: rolling 25.6%, smoke 8/32 = 25.0%.
- update 604: rolling 25.8%, NaN/OOM 없이 정상 중단.

## M6-R4-3 DAgger 최종 시도

Exact solver-only는 같은 Expert smoke에서 17/32 = 53.125%를 기록했다.
문제는 solver 상한이 아니라 solver trajectory teacher와 model 방문 상태의
distribution shift로 판단했다. 모델이 실제 방문한 Expert 상태를 round당
12,000개 수집하고 solver로 재라벨링하되 solver 출력은 inference 입력으로
사용하지 않는 DAgger 경로를 구현했다.

- Round 1: Expert 6/32 = 18.75%, Beginner 30/32, Intermediate 25/32.
- Round 2: Expert 7/32 = 21.875%, Beginner 30/32, Intermediate 24/32.
- Final held-out posterior: Brier 0.002164, NLL 0.414180,
  certain-safe precision 98.76%.
- policy loss는 각 round 안에서 감소했지만 Expert action ranking으로 전이되지 않았다.
- failed bootstrap-dagger checkpoint는 발행하지 않고 삭제한다.

R4-2 첫 production, 보존 강화 production, DAgger라는 서로 다른 세 방법이
Expert 40% 게이트를 통과하지 못했다. 설계 문서의 STOP 조건에 따라 추가 반복을
중단하며 새로운 대규모 아키텍처나 게이트 변경은 사용자 승인 없이는 진행하지 않는다.

## 불변식

- Expert 40% 게이트와 seed/game 수를 낮추지 않았다.
- solver 출력은 teacher label에만 사용했고 policy inference 입력에는 넣지 않았다.
- 실패한 모델을 M6 완료 checkpoint로 발행하지 않았다.
- M6 미완료 상태에서 M7로 진행하지 않는다.

## 다음 결정

현재 표현과 PPO objective를 유지한 추가 반복은 같은 정체를 반복할 가능성이 높다.
다음 단계는 Expert 전용 posterior/constraint 표현과 solver distillation을 포함하는
새 정책 아키텍처가 필요하다. 이는 하네스의 큰 아키텍처 변경에 해당하므로 사용자
승인 후 docs/M6_R4_설계제안.md의 고정 계약과 게이트로 별도 M6-R4를 진행한다.
