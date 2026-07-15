# M6 Expert 수렴 블로커

**갱신일**: 2026-07-15
**상태**: BLOCKED - Expert 40% 게이트 미달, 아키텍처 변경 승인 필요

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

## 불변식

- Expert 40% 게이트와 seed/game 수를 낮추지 않았다.
- solver 출력은 teacher label에만 사용했고 policy inference 입력에는 넣지 않았다.
- 실패한 모델을 M6 완료 checkpoint로 발행하지 않았다.
- M6 미완료 상태에서 M7로 진행하지 않는다.

## 다음 결정

현재 표현과 PPO objective를 유지한 추가 반복은 같은 정체를 반복할 가능성이 높다.
다음 단계는 Expert 전용 posterior/constraint 표현과 solver distillation을 포함하는
새 정책 아키텍처가 필요하다. 이는 하네스의 큰 아키텍처 변경에 해당하므로 사용자
승인 후 별도 M6-R4로 진행한다.
