# Minesweeper AI Speedrun

강화학습으로 "초보자" 신경망을 지뢰찾기 전문가로 학습시키고(Python + PyTorch 서버),
React 대시보드에서 실시간 관찰하며, Chrome 확장으로 minesweeper.online에서 시연한다.

> 설계 사양은 루트 문서를 단일 출처로 한다: `지뢰찾기AI_기술백서.md`,
> `지뢰찾기AI_디자인백서.md`, `지뢰찾기AI_파일트리.md`, `지뢰찾기AI_로컬환경_4060.md`.
> 진행 절차·DoD는 `지뢰찾기AI_하네스.md`, 진행 상태는 `PROGRESS.md`.

## 모노레포 구조

```
packages/
  core/            @msai/core — 게임 엔진 + RL 환경 + 논리 솔버 (의존성 0, no DOM/ONNX)
  agent/           @msai/agent — ONNX 추론 + 하이브리드 (예정)
  web/             @msai/web — React + Vite 대시보드 (예정)
  online-adapter/  @msai/online-adapter — Chrome MV3 확장 (예정)
server/            Python — FastAPI + PyTorch 학습/서빙 (예정)
```

## 개발 (JS/TS)

```bash
pnpm install
pnpm --filter @msai/core test      # vitest
pnpm -r exec tsc --noEmit          # 타입체크
pnpm lint                          # eslint (패키지 경계 룰 포함)
```

## 진행 상태

| Phase | 내용 | 상태 |
|---|---|---|
| M1 | `@msai/core` 엔진 (보드·캐스케이드·3BV·env) | ✅ 완료 |
| M2 | `@msai/core` 논리 솔버 (단일점·패턴·CSP·확률) | 진행 예정 |
| M3–M10 | 서버·학습·ONNX·web·확장·배포 | 예정 |

자세한 내용은 `PROGRESS.md` 참조.
