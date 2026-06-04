# 지뢰찾기 AI 스피드런 — 파일 트리 (단일 출처)

**버전**: 0.2
**작성일**: 2026년 06월 04일
**동반 문서**: `지뢰찾기AI_기술백서.md`, `지뢰찾기AI_디자인백서.md`

> 이 문서는 모노레포 구조의 **단일 출처(Single Source of Truth)** 다. Claude Code는 이 트리대로 스캐폴딩하고, 기술 백서 §1~§4와 본 문서 §5의 마일스톤 순서로 채워 넣는다.
> v0.1 대비 변경: 서버리스 → **Python 서버(FastAPI+PyTorch)**, JS 학습 → **Python 학습 1순위**, 프론트 → **React+Vite**, 브라우저/확장 추론 → **ONNX(onnxruntime-web)**.
> 경계 규칙: `agent`(추론)와 `online-adapter`(사이트 조작)는 서로 import하지 않는다. 공유는 `core`의 순수 타입/솔버만. 학습은 오직 `server/`에서 한다.

---

## 1. 전체 트리

```
minesweeper-ai/
├── CLAUDE.md                      # 레포 전역 가드레일(기술 백서 §7). 모든 에이전트가 먼저 읽음
├── README.md                      # 소개·로컬 실행(docker compose up)·구조 안내
├── docker-compose.yml             # 로컬 개발: server + postgres (+ 선택 web)
├── .env.example                   # 서버/DB 환경변수 템플릿(시크릿 비커밋)
├── .gitignore                     # node_modules, dist, __pycache__, *.pt, *.onnx, .venv
├── .github/
│   └── workflows/
│       ├── ci.yml                 # JS(lint/typecheck/test/build) + Python(ruff/pytest) + 확장 빌드
│       └── deploy.yml             # main: 서버 이미지 빌드/배포 + web 정적 배포 + 확장 zip
│
├── pnpm-workspace.yaml            # packages/* (JS 워크스페이스)
├── package.json                   # 루트(JS). 공통 스크립트
├── tsconfig.base.json             # 공통 TS 설정(strict). 각 패키지 extends
├── eslint.config.js               # 패키지 경계 import 금지 룰 포함
├── .prettierrc
│
├── packages/                      # ── JS/TS 영역 (클라이언트·공유) ──
│   ├── core/                      # @msai/core — 엔진 + RL 환경 + 논리 솔버 (의존성 0)
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── src/
│   │   │   ├── index.ts           # 공개 API 배럴
│   │   │   ├── types.ts           # Board, CellState, Difficulty, Action, StepResult
│   │   │   ├── rng.ts             # 시드 RNG(mulberry32) — 재현성
│   │   │   ├── board.ts           # 생성/첫클릭안전/오픈/캐스케이드(BFS)/승패
│   │   │   ├── difficulty.ts      # Beginner·Inter·Expert·Custom 프리셋
│   │   │   ├── metrics.ts         # 3BV, 3BV/s
│   │   │   ├── env.ts             # MinesweeperEnv: reset/step/legalActionMask/render
│   │   │   └── solver/
│   │   │       ├── index.ts       # solveStep: 단일점→패턴→CSP 통합 진입점
│   │   │       ├── single-point.ts
│   │   │       ├── patterns.ts    # 1-2-1, 1-2-2-1 등
│   │   │       ├── csp.ts         # 프론티어 열거 → 확실 안전/지뢰
│   │   │       └── probability.ts # 셀별 지뢰 확률(추측 폴백)
│   │   └── test/
│   │       ├── board.test.ts      # 캐스케이드/첫클릭안전/승패 (vitest)
│   │       ├── solver.test.ts     # NG 보드 ~100% 클리어 검증
│   │       └── metrics.test.ts
│   │
│   ├── agent/                     # @msai/agent — 추론 전용(ONNX) + 하이브리드 (학습 안 함)
│   │   ├── package.json           # deps: @msai/core, onnxruntime-web
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── index.ts
│   │       ├── encoding.ts        # Board → (H,W,C) 텐서 — server/trainer/encoding.py와 1:1
│   │       ├── onnx-session.ts    # onnxruntime-web 세션 로드/워밍업/추론
│   │       ├── masking.ts         # 비가용 셀 -∞ 마스킹
│   │       └── inference/
│   │           ├── policy.ts      # 순수 신경망 추론(목표 A 데모)
│   │           └── hybrid.ts      # core 솔버 + 모델 확률 결합(목표 B, 실전 기록)
│   │
│   ├── web/                       # @msai/web — React + Vite 대시보드(조종석)
│   │   ├── package.json           # deps: react, @msai/core, @msai/agent, @tanstack/react-query, zustand, (uplot|recharts)
│   │   ├── tsconfig.json
│   │   ├── vite.config.ts         # base 경로·worker·proxy(/api → server)
│   │   ├── index.html
│   │   └── src/
│   │       ├── main.tsx           # 엔트리(QueryClientProvider 등)
│   │       ├── App.tsx            # 셸 레이아웃·라우팅
│   │       ├── api/
│   │       │   ├── client.ts      # REST 클라이언트(서버 API)
│   │       │   ├── ws.ts          # WebSocket(메트릭 스트림) 구독
│   │       │   └── queries.ts     # TanStack Query 훅(models/runs/benchmarks)
│   │       ├── state/
│   │       │   └── uiStore.ts     # Zustand 로컬 UI 상태(선택 모델·추론 모드)
│   │       ├── views/
│   │       │   ├── Train.tsx      # 라이브 보드 + 실시간 메트릭 + ConfigForm
│   │       │   ├── Play.tsx       # ONNX+솔버 하이브리드 자동 플레이
│   │       │   ├── Benchmark.tsx  # N판 시행 리포트
│   │       │   └── Models.tsx     # 체크포인트 목록·내보내기/가져오기
│   │       ├── components/
│   │       │   ├── layout/        # Sidebar, Header, Footer
│   │       │   ├── ui/            # Button, Card, Badge, Modal (원자)
│   │       │   └── features/
│   │       │       ├── BoardView.tsx     # 보드 + 하이라이트(안전/지뢰/추측)
│   │       │       ├── MetricChart.tsx   # 실시간 라인 차트(채움 그라데이션 금지)
│   │       │       ├── ConfigForm.tsx    # TrainConfig 폼·검증
│   │       │       ├── RunControls.tsx
│   │       │       ├── ModelList.tsx
│   │       │       └── BenchmarkTable.tsx
│   │       └── styles/
│   │           ├── global.css
│   │           └── tokens.css     # 디자인 토큰(디자인 백서 §5)
│   │
│   └── online-adapter/            # @msai/online-adapter — Chrome MV3 확장(사이드로드)
│       ├── package.json
│       ├── tsconfig.json
│       ├── vite.config.ts         # @crxjs/vite-plugin 등 MV3 빌드
│       ├── CLAUDE.md              # 어댑터 전용 가드레일(스크래핑 금지·레이트 제한·권한 최소)
│       ├── public/
│       │   ├── manifest.json      # MV3, host_permissions: minesweeper.online 게임 경로 한정
│       │   └── icons/
│       └── src/
│           ├── content/
│           │   ├── index.ts       # 콘텐츠 스크립트 엔트리(주입)
│           │   ├── dom-contract.ts# 셀렉터/클래스 상수 단일 출처(사이트 변경 시 여기만)
│           │   ├── read-board.ts  # DOM → @msai/core Board 파싱
│           │   ├── dispatch.ts    # 좌/우/코드 클릭 이벤트 합성
│           │   └── loop.ts        # 관찰→hybrid 판단→실행 루프(won/lost까지)
│           ├── background/
│           │   └── service-worker.ts
│           └── popup/
│               ├── popup.html
│               └── popup.ts       # 시작/정지·모드·모델 선택, 클리어 시간 표기
│
├── server/                        # ── Python 영역 (학습 1순위 + API + 서빙) ──
│   ├── pyproject.toml             # 의존성(또는 requirements.txt)
│   ├── Dockerfile                 # FastAPI + PyTorch 런타임(GPU 베이스 선택)
│   ├── app/
│   │   ├── main.py                # FastAPI 엔트리(REST + WebSocket 라우팅)
│   │   ├── api/
│   │   │   ├── training.py        # 학습 작업 시작/정지/재개/조회
│   │   │   ├── models.py          # 모델 목록/조회/업로드/내보내기
│   │   │   ├── benchmark.py       # 벤치 실행/결과 조회
│   │   │   └── leaderboard.py     # (선택) 리더보드
│   │   ├── ws/
│   │   │   └── metrics.py         # 학습 메트릭 스트림(배치/스로틀)
│   │   ├── core/
│   │   │   ├── config.py          # 설정(.env 로드)
│   │   │   └── deps.py            # 인증·의존성
│   │   ├── db/
│   │   │   ├── session.py         # SQLAlchemy 세션
│   │   │   ├── models.py          # ORM(TrainingRun, Model, BenchmarkResult ...)
│   │   │   └── migrations/        # Alembic
│   │   ├── storage/
│   │   │   └── weights.py         # 가중치 파일/객체저장 추상화(.pt/.onnx)
│   │   └── schemas/               # pydantic — TS @msai/core 타입과 계약 일치
│   │       ├── training.py
│   │       ├── model.py
│   │       └── benchmark.py
│   ├── trainer/                   # ★ RL 학습 핵심(PyTorch) — 1순위 경로
│   │   ├── env.py                 # @msai/core 환경과 1:1 재현(규칙 동일)
│   │   ├── encoding.py            # @msai/agent encoding.ts와 1:1(채널 동일)
│   │   ├── model.py               # fully-conv CNN(+dueling)
│   │   ├── replay.py              # Prioritized Experience Replay
│   │   ├── dqn.py                 # Double DQN(타깃 네트워크·n-step)
│   │   ├── reward.py              # 보상 셰이핑
│   │   ├── curriculum.py          # Beginner→Inter→Expert 승급 게이트
│   │   ├── train.py               # 학습 루프(에피소드·체크포인트·메트릭 콜백)
│   │   ├── evaluate.py            # 승률/시간/3BV·s 벤치
│   │   └── export.py              # torch.onnx.export → ONNX(브라우저/확장 호환)
│   └── tests/
│       ├── test_env.py            # Python 엔진이 core 규칙과 동일한지
│       └── test_encoding_parity.py# 고정 보드 입력에 대해 TS와 인코딩 일치 검증
│
└── docs/
    ├── 지뢰찾기AI_기술백서.md
    ├── 지뢰찾기AI_디자인백서.md
    └── 지뢰찾기AI_파일트리.md       # 본 문서
```

---

## 2. 영역·패키지별 책임 요약

| 영역/패키지 | 한 줄 책임 | 절대 하지 않는 것 |
|---|---|---|
| `@msai/core` (TS) | 게임 규칙·환경·논리 솔버. 의존성 0, 결정론적 | ONNX/TF 의존, DOM 접근 금지 |
| `@msai/agent` (TS) | ONNX 추론 + 하이브리드(솔버+모델) | **학습 금지**, DOM 접근 금지 |
| `@msai/web` (React) | 학습 관찰·플레이·벤치 조종석 | 학습 수행 금지(서버에 위임) |
| `@msai/online-adapter` (MV3) | minesweeper.online 플레이(읽기·클릭) | 학습 코드 import·사용자 데이터 수집 금지 |
| `server/app` (Python) | API·WS·DB·서빙 | — |
| `server/trainer` (Python) | ★ RL 학습 + ONNX 내보내기 | JS와 다른 아키텍처/인코딩 금지(호환 깨짐) |

> 언어 간 이중 구현(필수·주의): 게임 규칙은 `core`(TS)와 `server/trainer/env.py`(Python)에, 상태 인코딩은 `agent/encoding.ts`와 `server/trainer/encoding.py`에 각각 존재. `server/tests/test_encoding_parity.py`로 일치 보장.

---

## 3. 핵심 설정 파일 스케치

### 3.1. `docker-compose.yml` (요지)
```yaml
services:
  db:
    image: postgres:16
    environment: { POSTGRES_DB: msai, POSTGRES_PASSWORD: <env> }
    volumes: ["pgdata:/var/lib/postgresql/data"]
  server:
    build: ./server
    env_file: .env
    depends_on: [db]
    ports: ["8000:8000"]
    # GPU 사용 시 deploy.resources.reservations.devices 설정
volumes: { pgdata: {} }
```

### 3.2. `pnpm-workspace.yaml`
```yaml
packages:
  - "packages/*"
```
> Python(`server/`)은 pnpm 워크스페이스가 아니라 별도 venv/poetry로 관리.

### 3.3. 루트 `package.json` (발췌)
```json
{
  "name": "minesweeper-ai", "private": true,
  "scripts": {
    "dev:web": "pnpm --filter @msai/web dev",
    "build": "pnpm -r build",
    "test": "pnpm -r test",
    "typecheck": "pnpm -r exec tsc --noEmit",
    "lint": "eslint .",
    "build:ext": "pnpm --filter @msai/online-adapter build"
  }
}
```

### 3.4. `online-adapter/public/manifest.json` (요지, MV3)
```json
{
  "manifest_version": 3,
  "name": "Minesweeper AI Adapter",
  "version": "0.1.0",
  "host_permissions": ["https://minesweeper.online/*"],
  "background": { "service_worker": "service-worker.js" },
  "content_scripts": [
    { "matches": ["https://minesweeper.online/*/game/*", "https://minesweeper.online/*/new-game*"],
      "js": ["content.js"] }
  ],
  "action": { "default_popup": "popup.html" }
}
```
> 실제 매치 패턴·셀렉터는 DevTools로 검증 후 확정(기술 백서 §7).

---

## 4. 컨벤션
- **언어**: TS(`strict`) + Python 3.11+(타입힌트·ruff·pytest). 식별자·파일명 영문.
- **계약 일치**: 서버 pydantic 스키마 ↔ TS 타입을 수동/생성 동기화. 인코딩·규칙 패리티 테스트 필수.
- **import 경계**: eslint `no-restricted-imports`로 `agent↔online-adapter` 상호 참조와 `core`의 ONNX/DOM 의존 차단.
- **산출물**: 대용량 가중치(`*.pt`, `*.onnx`)는 git 비커밋. 공유는 서버 모델 API 또는 릴리스 아티팩트.
- **확장**: Web Store 미배포, 사이드로드. CI는 zip 아티팩트만 생성.
- **디자인 토큰**: `web/src/styles/tokens.css` 단일 관리(디자인 백서 §5 anti-cliché 준수).

---

## 5. 구현 착수 지점 (Claude Code 마일스톤)
1. **M1 — `core` 엔진**: types→board→env→metrics, vitest. (토대)
2. **M2 — `core` 솔버**: 단일점→패턴→CSP→확률. NG 보드 ~100% 클리어 검증.
3. **M3 — `server` 골격**: FastAPI + DB(ORM/마이그레이션) + 스키마 + 헬스/모델 API.
4. **M4 — `server/trainer`**: env.py(규칙)·encoding.py·model.py·replay·dqn·train. Beginner 학습 곡선 우상향.
5. **M5 — 메트릭 스트림**: 학습 메트릭 WebSocket + 체크포인트 저장.
6. **M6 — 커리큘럼**: Beginner→Inter→Expert 전이 학습.
7. **M7 — ONNX export + `agent`**: export.py → onnxruntime-web 추론 + encoding 패리티 테스트.
8. **M8 — `web` 대시보드(React)**: Train/Play/Benchmark/Models, 실시간 차트, 하이브리드 플레이.
9. **M9 — `online-adapter`(MV3)**: dom-contract 검증·보드 파싱·클릭·루프. NG 모드 우선 시연.
10. **M10 — 벤치·리더보드 + 배포**: BenchmarkResult, Actions, 서버 이미지·web 정적·확장 zip.

> 각 마일스톤 완료 시 해당 게이트(테스트·승률·패리티)를 통과시키고 다음으로 진행.
