# 지뢰찾기 AI 스피드런 기술 백서 (Technical Whitepaper)

**버전**: 0.2
**작성일**: 2026년 06월 04일
**작성자**: 1인 개발 (Bryan)
**참고 문서**: 디자인 백서 v0.2, 파일 트리 v0.2, minesweeper.online DOM 계약

> 본 문서는 첨부된 "웹 프로젝트용 기술 백서 템플릿"의 절 구조를 따른다. 다만 이 프로젝트는 단순 웹앱이 아니라 강화학습 서버 + 클라이언트 + 브라우저 확장이 결합된 형태라, 템플릿에 없는 학습 파이프라인(§2.5)과 RL 상세(§4.3) 등을 추가하고 일부 절의 정의를 ML 맥락으로 수정했다.
> 이 문서는 사람이 줄 단위로 읽기보다 **Claude Code가 파싱해 모노레포를 구현**하기 위한 사양서다. 파일 구조의 단일 출처는 별도 문서 `지뢰찾기AI_파일트리.md`다.

---

## 1. 프로젝트 개요 (Project Overview)

### 1.1. 프로젝트 명
**Minesweeper AI Speedrun** (가칭 `minesweeper-ai`)

### 1.2. 목적 (Purpose)
- 아무것도 모르는 "초보자" 신경망을 강화학습으로 점진 훈련시켜, 지뢰찾기를 사람 수준 이상으로 풀게 만든다.
- 학습은 **Python(PyTorch) 서버**에서 수행하고, 진행 과정을 **React 대시보드**에서 실시간 관찰하며, 학습된 모델을 **Chrome 확장**으로 minesweeper.online에서 시연·기록 측정한다.
- 설계 전제(수학적 사실): 표준 모드 상급(Expert, 30×16, 지뢰 99)은 강제 50/50 추측 때문에 **완벽한 플레이로도 승률 상한이 ~40%**다. 따라서 "100% 클리어 봇"은 표준 모드에서 불가능하며, 일관된 시연·기록은 순수 논리로 풀리는 **추측 없이 모드(NG)** 에서 수행한다.

### 1.3. 핵심 차별점 (Key Differentiators)
1. **이중 엔진 분리(학습 ↔ 실전)**: "아무것도 모르는 AI → 전문가"라는 서사는 순수 RL 에이전트가 담당(데모/학습 곡선)하고, 실제 기록은 논리 솔버를 1차로·신경망 확률을 추측 보조로 쓰는 **하이브리드 추론**이 담당한다. 같은 가중치를 공유하되 추론 방식만 다르다.
2. **크기 불변 전이학습**: 전결합층 없는 **fully-convolutional CNN**으로 입력 보드 크기에 무관하게 동작시켜, 초급에서 배운 가중치를 상급으로 전이하는 커리큘럼(Beginner→Intermediate→Expert)을 구현한다.
3. **서버 중심 안정성**: PyTorch GPU 학습 + PostgreSQL 영속화로 장시간 학습을 안정적으로 운용하고, 학습 결과를 **ONNX로 내보내** 브라우저·확장에서 onnxruntime-web으로 저지연 추론한다(클라이언트는 학습하지 않음).

---

## 2. 상세 기능 요구사항 (Detailed Requirements)

### 2.1. 시스템 환경 및 인터페이스 (System & Interface)
- **뷰 모드 (View Mode)**: 대시보드는 **Desktop First**. 데이터 밀도가 높은 계측기형 UI(보드 + 다중 차트 동시 표시)가 주 사용 시나리오. 모바일은 학습 모니터링용 읽기 전용 요약으로 축약.
- **테마 정책 (Theme Policy)**: CSS Variables 기반 **다크 우선**(계측기 감성) + 라이트 토글. 색·타이포 토큰은 디자인 백서 §5와 동기화.
- **실행 토폴로지(추가)**:
  - 학습 서버: Python 3.11+ / FastAPI / PyTorch (GPU 권장). 장시간 프로세스.
  - 웹 클라이언트: 브라우저(React). 서버와 REST + WebSocket으로 통신.
  - 외부 플레이: Chrome MV3 확장(minesweeper.online에 콘텐츠 스크립트 주입).
  - DB: PostgreSQL(메타/리더보드), 가중치 파일은 객체 저장소 또는 볼륨.

### 2.2. 사용자 상호작용 로직 (Interaction Logic)
- **이벤트 처리 (Event Handling)**:
  - **Input**: 학습 설정 폼(`TrainConfig`) 입력은 슬라이더/숫자 입력 + 클라이언트 범위 검증. 학습 시작/일시정지/재개/중단 버튼.
  - **Action**: 버튼 클릭 → 서버 학습 API 호출 → 작업 상태 변경 → **WebSocket으로 진행 메트릭 푸시 수신** → 차트 갱신. 학습은 장시간이라 낙관적 UI를 쓰지 않고 **서버 상태를 진실 원천으로 동기화**한다.
- **데이터 검증 (Validation)**:
  - 클라이언트: 범위/형식(에피소드>0, 학습률·γ 범위, 난이도 enum).
  - 서버: pydantic 스키마 검증 + 정책(동시 실행 학습 작업 수 제한, 리소스 가드).

### 2.3. 데이터 모델 (Data Model)
주요 엔티티 스키마(서버 DB 기준, TS/pydantic 양쪽 타입 일치):
1. **TrainingRun**: `id(UUID)`, `difficulty(Enum)`, `config(JSON: TrainConfig)`, `status(Enum: queued|running|paused|done|failed)`, `startedAt`, `endedAt`, `summary(JSON: 최종 승률 등)`.
2. **Model(Checkpoint)**: `id(UUID)`, `runId(FK)`, `episode(Int)`, `winRate(Float)`, `archHash(String)`, `weightsUri(String)`, `format(Enum: pt|onnx)`, `createdAt`.
3. **BenchmarkResult**: `id(UUID)`, `modelId(FK)`, `mode(Enum: standard|ng)`, `difficulty(Enum)`, `games(Int)`, `winRate(Float)`, `avgClearMs(Float)`, `avg3BVps(Float)`, `guessRate(Float)`, `seed(Int?)`.
4. **(선택) LeaderboardEntry**: `id`, `modelId`, `mode`, `metricValue`, `createdAt`.

### 2.4. 출력 및 성능 기준 (Output & Performance)
- **결과물 형식**: 학습 체크포인트(`.pt`), 추론용(`.onnx`), 메트릭 스트림(WS JSON), 벤치 리포트(JSON), 모델 내보내기/가져오기.
- **품질 기준 (QA Standards)**:
  - 학습 안정성: 커리큘럼 단계별 승률 게이트 통과, 상급 학습 곡선이 ~40% 근방 수렴(상한 인지).
  - 추론 지연: 브라우저 ONNX 추론 한 수 수 ms~수십 ms, 하이브리드 의사결정 한 판 합 수백 ms 이하(클릭 실행 제외).
  - 대시보드 메트릭 갱신 지연: 200ms 이하 체감.
  - 브라우저 호환: 웹은 Chrome/Safari/Edge 최신, 확장은 Chrome(MV3).

### 2.5. 학습 파이프라인 (Training Pipeline) — *템플릿 추가 절*
서버 측 RL 학습 1사이클:
```
reset(난이도/시드) → 관찰 인코딩 → ε-greedy 행동(마스킹) → step(보상/종료)
   → 경험을 PER 버퍼에 저장 → 배치 샘플 → Double DQN 손실 → 파라미터 갱신
   → 주기적 target network 동기화 → N 에피소드마다 체크포인트(.pt) 저장·DB 기록
   → 평가(승률 이동평균) → 커리큘럼 승급 판정 → (학습 종료 시) ONNX export
```
- 진행 메트릭(에피소드, 승률 이동평균, 손실, ε, 초당 스텝, 현재 난이도)은 WebSocket으로 대시보드에 스트리밍.

---

## 3. 기술 스택 및 라이브러리 (Tech Stack)

### 3.1. Core
- **Frontend**: React 18 + TypeScript 5 + Vite 5.
- **Backend**: Python 3.11+ / FastAPI(REST + WebSocket) / PyTorch(학습) / Uvicorn(+Gunicorn).
- **Database**: PostgreSQL 16 (메타·벤치·리더보드). 가중치 파일은 객체 저장소(S3 호환) 또는 마운트 볼륨.

### 3.2. Libraries & Tools
1. **PyTorch** (필수)
   - **용도**: 강화학습 핵심(모델·학습 루프). RL 생태계 성숙도·디버깅 용이성·ONNX 내보내기 때문에 채택.
   - **설정 값**: AMP(혼합정밀, GPU 시), `torch.onnx.export`로 추론 그래프 내보내기.
2. **onnxruntime-web** (필수)
   - **용도**: 브라우저·확장에서 내보낸 ONNX 모델 추론. WebGL/WASM 백엔드.
3. **TanStack Query (React Query)** (권장)
   - **용도**: 서버 상태(학습 작업/모델/벤치) 캐싱·동기화·재시도·로딩/에러 처리.
4. **Zustand** (선택)
   - **용도**: 클라이언트 로컬 UI 상태(토글·선택값).
5. **차트 라이브러리** (필수)
   - **용도**: 메트릭 시각화. 고빈도 스트림은 **uPlot**, 일반 리포트는 recharts 중 택1.
6. **FastAPI + pydantic / SQLAlchemy + Alembic** (필수)
   - **용도**: API·스키마 검증·ORM·마이그레이션. pydantic 스키마는 TS `@msai/core` 타입과 **계약 일치** 유지.
7. **@crxjs/vite-plugin** (선택)
   - **용도**: MV3 확장 빌드(Vite 기반).

---

## 4. 아키텍처 및 로직 (Architecture & Logic)

### 4.1. 상태 관리 전략 (State Management)
- **Scope**:
  - **서버 상태**(학습 작업·모델·벤치 결과): 진실 원천은 서버. 프론트는 TanStack Query로 캐시/동기화.
  - **실시간 학습 진행**: WebSocket 구독으로 푸시 갱신(쿼리 캐시 무효화/병합).
  - **클라이언트 로컬 상태**(UI 토글·현재 선택 모델·추론 모드): Zustand.
- **Tool**: TanStack Query(서버 상태) + Zustand(로컬). WebSocket 클라이언트는 별도 모듈.

```typescript
// 로컬 UI 스토어 스케치 (Zustand)
const useUI = create((set) => ({
  selectedModelId: null,
  inferMode: 'hybrid',          // 'policy' | 'hybrid'
  setModel: (id) => set({ selectedModelId: id }),
  setMode: (m) => set({ inferMode: m }),
}));
```

### 4.2. 주요 동작 파이프라인 (Main Workflow)
1. **초기화 (Init)**: 웹 로드 → 서버 헬스 체크 → 모델 목록 조회. Play 모드 진입 시 선택 모델의 ONNX를 onnxruntime-web으로 로드(워밍업 1회).
2. **학습 (Process)**: `TrainConfig` 제출 → 서버가 `TrainingRun` 생성·시작 → 메트릭 WS 스트림 → 대시보드 갱신 → 체크포인트 생성 시 모델 목록 갱신.
3. **추론/플레이 (Update)**: 모델 선택 → (브라우저) ONNX 추론 + `@msai/core` 솔버를 결합한 하이브리드로 한 판 진행 → BoardView가 확실/추측 셀을 색으로 구분하며 시각화.

### 4.3. 핵심 알고리즘 (Core Algorithms) — *ML 프로젝트의 중심*
- **게임 엔진 규칙**(`@msai/core` & `server/trainer`): 시드 RNG, 첫 클릭 안전 배치, 0 영역 BFS 캐스케이드, 승패 판정, 3BV/3BV·s 산출. (TS·Python 양쪽에 동일 규칙 구현 — §7 주의사항 참조)
- **논리 솔버**(`@msai/core/solver`):
  1. 단일점(single-point): 숫자=인접 플래그 → 나머지 안전 / 숫자=인접 hidden → 전부 지뢰.
  2. 패턴(1-2-1, 1-2-2-1 등 부분집합 규칙).
  3. CSP 프론티어 열거: 제약 만족 지뢰 배치를 열거(분할/근사)해 확실 안전·확실 지뢰 도출, 셀별 지뢰 확률 산출.
- **강화학습**(`server/trainer`):
  - 알고리즘: **Double DQN + Dueling + Prioritized Experience Replay + n-step return**. ε-greedy(1.0→0.01 감쇠), target network 주기 동기화.
  - 상태 인코딩: 보드를 `(H, W, C)` 다채널 텐서로(채널: hidden, flag, 숫자0~8 원-핫, 선택 채널: 잔여 지뢰 정규화·프론티어 마스크).
  - 행동: hidden 셀 선택(H×W). **행동 마스킹 필수**(비가용 셀 Q값 -∞).
  - 보상: 안전 오픈 +, 승리 ++, 지뢰 --, 선택적 스텝 페널티(속도 유도).
  - 모델: fully-conv CNN(잔차 블록 ×4~6, `padding=same`, 1×1 conv로 셀별 출력) + 선택적 Dueling 분기.
  - 커리큘럼: Beginner 9×9·10 → Intermediate 16×16·40 → Expert 30×16·99. 단계별 "최근 N판 승률 게이트" 통과 시 승급, 가중치 승계.
- **하이브리드 추론**(`@msai/agent/inference/hybrid`): 솔버가 확정 수(안전/지뢰)를 주면 그대로 실행, 강제 추측 시에만 CSP 확률과 신경망 mine-probability를 결합해 최소 위험 셀 선택. NG 모드에서는 거의 매 턴 확정 수가 나와 ~100% 승률·고속.

---

## 5. UI 구현 가이드 (Implementation Guide)
*상세 시각 규칙은 디자인 백서 참조. 여기서는 코드 연동 토큰·공통 컴포넌트만.*

### 5.1. 디자인 토큰 (Design Tokens)
- **Colors**: `--bg`(#0e1116), `--surface`(#161b22), `--border`(#2a313c), `--text`(#e6edf3), `--text-muted`(#8b949e), `--accent`(#4c8dff), `--safe`(#2ea043), `--danger`(#d2453f), `--guess`(#d2a23f).
- **Typography**: UI `system-ui`/Inter, 수치 `JetBrains Mono`(또는 ui-monospace). Base 14px.
- **Breakpoints**: Mobile(≤640px), Tablet(≤1024px), Desktop(>1024px). Desktop 우선.

### 5.2. 공통 컴포넌트 (Shared Components)
- **BoardView**: props `board`, `highlight`(safe/mine/guess 맵). 숫자 셀은 고전 지뢰찾기 색 차용(1 청, 2 녹, 3 적 …), 주변 UI는 절제.
- **MetricChart**: props `series`, `window`(이동평균 구간). 그라데이션 채움 금지.
- **ConfigForm**: props `config`, `onSubmit`. 범위 검증 내장.
- **RunControls / ModelList / BenchmarkTable**.
- **Modal**: React Portal, z-index 토큰으로 관리.

---

## 6. 파일 구조 (File Structure)
요약만 표기하며, **전체 트리의 단일 출처는 별도 문서** `지뢰찾기AI_파일트리.md`.
```text
minesweeper-ai/
├── packages/            # JS/TS (pnpm workspaces)
│   ├── core/            # @msai/core — 엔진 + RL 환경 + 논리 솔버 (의존성 0)
│   ├── agent/           # @msai/agent — ONNX 추론 + 하이브리드 (학습 안 함)
│   ├── web/             # @msai/web — React + Vite 대시보드
│   └── online-adapter/  # @msai/online-adapter — Chrome MV3 확장
├── server/              # Python — FastAPI + PyTorch (학습 1순위)
│   ├── app/             # API·WS·DB·schemas
│   └── trainer/         # RL 학습 핵심 + ONNX export
├── db/                  # 스키마/마이그레이션
└── docs/                # 기술/디자인/파일트리 백서
```

---

## 7. 개발 시 주의사항 (Implementation Notes)
1. **보안 (Security)**:
   - 서버 API 인증 토큰 필수, CORS는 웹·확장 출처로 한정. 입력 살균·스키마 검증.
   - 시크릿(.env)·DB 자격증명 코드 비포함. URL 쿼리스트링에 민감정보 금지.
2. **성능 최적화 (Optimization)**:
   - 메트릭 WS 스트림은 서버에서 배치/스로틀(예: 100ms 묶음). GPU 메모리 관리(AMP·배치 크기).
   - 브라우저 ONNX 추론 워밍업 1회 후 사용. React 메모이제이션·코드 스플리팅(라우트 단위).
3. **이슈 대응 (Known Issues)**:
   - **언어 간 규칙·인코딩 이중 구현**: 게임 규칙은 `@msai/core`(TS)와 `server/trainer/env.py`(Python)에, 상태 인코딩은 `@msai/agent/encoding.ts`와 `server/trainer/encoding.py`에 **각각** 존재한다(학습=Python, 클라이언트 추론=TS). 둘이 어긋나면 추론이 붕괴하므로, **고정 보드 입력에 대한 인코딩 출력 일치 테스트**를 양쪽에 둔다.
   - **minesweeper.online DOM 변경**: 셀렉터를 `online-adapter`의 `dom-contract.ts` 한 파일에 격리, DevTools로 검증 후 사용.
   - **약관·페어플레이**: 자동화(봇) 기록의 공개 순위 제출은 제한될 수 있음. 기본 용도는 개인 벤치·시연. 사이트 정책 준수.
   - **승률 상한**: 표준 상급 ~40% 한계. 일관 시연은 NG 모드.
   - **서버리스 아님(변경됨)**: 학습은 타임아웃 없는 장시간 서버 프로세스(GPU 권장). 서버는 영속화·학습·서빙을 모두 담당.

---

### 부록 A. 참고 (성능 수치 출처)
- 알고리즘 솔버 상급 승률 30%대 후반~40% 초반(사실상 상한): AlexMGitHub/Minesweeper-DDQN, mrgriscom(10만 판 37.8%), DavidNHill(~40%).
- 소형 6×6/지뢰4 DQN ~93.3%, SL ~91.2%: MDPI Applied Sciences 2025, 15, 2490.
- 16×16 DQN 밀도별 ~15~45%: arXiv 2102.06019.
> 위 수치는 기대치 설정용 참고이며, 본 프로젝트 자체 벤치(§2.4·BenchmarkResult)로 재측정해 갱신한다.
