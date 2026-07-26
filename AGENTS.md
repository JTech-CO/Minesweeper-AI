# AGENTS.md — 작업 규칙 (루트, 매 세션 로드)

이 파일은 **항상 짧게 유지한다.** 상세 절차·런북은 `docs/지뢰찾기AI_하네스.md`, 진행 상태는 `PROGRESS.md`에 있다.

## 0. 프로젝트
Minesweeper AI Speedrun — 강화학습으로 지뢰찾기를 학습시키고(서버: Python+PyTorch), React 대시보드에서 관찰하며, Chrome 확장으로 minesweeper.online에서 시연한다.

문서 인덱스(단일 출처):
- 기술 사양: `docs/지뢰찾기AI_기술백서.md`
- 디자인 사양: `docs/지뢰찾기AI_디자인백서.md`
- 파일 구조(SSoT): `docs/지뢰찾기AI_파일트리.md`
- 로컬 GPU 환경: `docs/지뢰찾기AI_로컬환경_4060.md`
- **진행 절차·DoD·런북: `docs/지뢰찾기AI_하네스.md`**
- **진행 상태(세션 인계): `PROGRESS.md`**

## 1. 세션 프로토콜 (매번)
1. **시작**: `PROGRESS.md`를 먼저 읽는다 → 현재 phase·다음 할 일·미결 질문 파악. 해당 phase의 하네스 절을 확인한다.
2. **작업**: 한 번에 **한 phase만**. 그 phase의 DoD 게이트를 통과하기 전에는 다음 phase로 넘어가지 않는다.
3. **종료**: `PROGRESS.md`를 갱신한다(완료 체크·다음 할 일·새 미결 질문·결정 로그). 그 후 커밋.

## 2. 명령 빠른참조
```bash
# JS/TS
pnpm install
pnpm -r test            # vitest (core 등)
pnpm -r exec tsc --noEmit
pnpm lint
pnpm --filter @msai/web dev
pnpm --filter @msai/online-adapter build

# DB + 서버 (로컬환경 가이드 참조)
docker compose up -d db
cd server && source .venv/bin/activate   # Windows: .venv\Scripts\activate
alembic upgrade head
uvicorn app.main:app --reload --port 8000
pytest

# 학습 (CUDA torch는 index-url로 별도 설치, 가이드 §2)
python -m trainer.train --difficulty beginner --episodes 100000
```

## 3. Phase 게이트 요약 (상세는 하네스)
| Phase | 완료 게이트(요지) |
|---|---|
| M1 core 엔진 | vitest 그린 + 결정론(시드)·첫클릭안전·캐스케이드·3BV 정확 |
| M2 core 솔버 | 확정 판정 거짓양성 0 + NG 보드 솔버 단독 ~100% 클리어 |
| M3 server 골격 | `alembic upgrade head` + `GET /health` 200 + 모델 메타 CRUD |
| M4 trainer | **Beginner 평가 승률 게이트(예 0.85) 도달** + loss 비발산 + 체크포인트 재현 |
| M5 메트릭 | 학습 메트릭 WS 수신 + 체크포인트 DB 기록 |
| M6 커리큘럼 | Beginner→Inter→Expert 자동 승급 (Expert ~40% 수렴은 정상) |
| M7 ONNX+agent | **인코딩 패리티 통과 + ONNX↔torch 수치 일치(1e-3 내)** + NG 하이브리드 ~100% |
| M8 web | 학습 시작→실시간 차트 + 플레이 시각화 + 디자인 토큰/anti-cliché 준수 |
| M9 adapter | 실제 사이트 보드 파싱 일치 + 클릭 동작 + NG 한 판 자동 클리어 |
| M10 배포 | 벤치 저장·표시 + CI 그린 + 배포 산출물 |

## 4. 멈춤 규칙 (STOP — 루프 돌지 말 것)
다음이면 즉시 멈추고 `PROGRESS.md`에 증상·시도·가설을 적은 뒤 사용자에게 보고·질문한다.
- 같은 에러/테스트 실패를 **서로 다른 방법으로 3회** 시도해도 못 고침.
- DoD 게이트(특히 M4 승률·M7 패리티)를 못 넘는데 **우회하고 싶을 때**.
- **불변식(§5)을 깨야만** 통과시킬 수 있을 때.
- 큰 아키텍처 변경이 필요해 보일 때(예: 패리티가 어려우니 추론도 서버로 옮기기) → 임의 변경 금지, 승인 요청.
- 외부 제약(GPU 없음, 사이트 DOM 접근 불가, 약관 위반 소지).

## 5. 불변식 (절대 위반 금지)
- **테스트·게이트 무결성**: 테스트를 삭제·약화하거나 게이트 수치를 임의로 낮춰 "통과"시키지 않는다.
- **패리티 항상 통과(M7)**: 인코딩(TS↔Python)·ONNX↔torch 수치 일치. 깨진 채 진행 금지.
- **이중 구현 동기화**: 게임 규칙은 `@msai/core`(TS)와 `server/trainer/env.py`(Python), 인코딩은 `agent/encoding.ts`와 `server/trainer/encoding.py`에 동일 유지.
- **패키지 경계**: `agent`↔`online-adapter` 상호 import 금지. `core`는 ONNX/DOM 의존 금지. **학습은 `server/`에서만**.
- **가드레일 약화 금지**: 스크래핑 금지·요청 레이트 제한·권한 최소(상세 `online-adapter/AGENTS.md`). 자동화 기록의 공개 순위 제출은 사이트 정책 준수.
- **비커밋**: 대용량 가중치(`*.pt`, `*.onnx`)와 시크릿(`.env`)은 git에 올리지 않는다.
- **상한 인지**: 표준 모드 Expert 승률 **~40%는 정상 상한**이다(버그 아님, 무한 튜닝 금지). 일관 시연은 NG 모드.

## 6. 작업 자세
- 추측하지 말고 해당 백서 절을 근거로 한다. 불확실하면 멈춤 규칙(§4)을 따른다.
- 디자인은 디자인 백서의 anti-cliché 원칙 준수(네온·다중 그라데이션·글래스모피즘 도배·장식 이모지 헤더 금지).
