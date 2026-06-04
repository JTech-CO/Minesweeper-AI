# Minesweeper AI Speedrun

강화학습으로 "초보자" 신경망을 지뢰찾기 전문가로 학습시키고(Python + PyTorch 서버),
React 대시보드에서 실시간 관찰하며, Chrome 확장으로 minesweeper.online에서 시연한다.

> 설계 사양은 `docs/`의 문서를 단일 출처로 한다: `docs/지뢰찾기AI_기술백서.md`,
> `docs/지뢰찾기AI_디자인백서.md`, `docs/지뢰찾기AI_파일트리.md`, `docs/지뢰찾기AI_로컬환경_4060.md`.
> 진행 절차·DoD는 `docs/지뢰찾기AI_하네스.md`, 진행 상태는 `PROGRESS.md`.

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

## 개발 (서버, Python)

로컬 개발은 기본 SQLite(무설치). CI/운영은 Postgres 16(`docker compose up -d db`).
학습(M4)에는 CUDA PyTorch가 필요하며 Python 3.11/3.12 권장(docs/지뢰찾기AI_로컬환경_4060.md).

```bash
cd server
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
alembic upgrade head                # 테이블 생성 (dev: sqlite:///./msai.db)
uvicorn app.main:app --reload --port 8000
pytest                              # /health + 모델 CRUD
ruff check . && ruff format --check .
```

## 진행 상태

| Phase | 내용 | 상태 |
|---|---|---|
| M1 | `@msai/core` 엔진 (보드·캐스케이드·3BV·env) | ✅ 완료 |
| M2 | `@msai/core` 논리 솔버 (단일점·부분집합·CSP·확률·NG 생성) | ✅ 완료 |
| M3 | `server` 골격 (FastAPI + SQLAlchemy + Alembic + /health·모델 CRUD) | ✅ 완료 |
| M4 | `server/trainer` 학습 (Double DQN, 커리큘럼) — torch | 진행 예정 |
| M5–M10 | 메트릭·커리큘럼·ONNX·web·확장·배포 | 예정 |

자세한 내용은 `PROGRESS.md` 참조.
