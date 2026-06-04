# 지뢰찾기 AI 스피드런 디자인 백서 (Design Whitepaper)

**버전**: 0.2
**작성일**: 2026년 06월 04일
**작성자**: 1인 개발 (Bryan)
**참고 문서**: 기술 백서 v0.2, 파일 트리 v0.2

> 본 문서는 첨부된 "웹 프로젝트용 디자인 백서 템플릿"의 절 구조를 따른다. 이 프로젝트의 UI는 일반 콘텐츠 사이트가 아니라 **학습 서버에 붙는 계측기형 대시보드 + 지뢰찾기 보드 시각화**라, 데이터 모듈 정의(§2.3)와 핵심 컴포넌트(§4.3)를 그 맥락으로 수정했다. 색·토큰 값은 기술 백서 §5.1과 동기화한다.

---

## 1. 프로젝트 개요 (Project Overview)

### 1.1. 프로젝트 명
**Minesweeper AI Speedrun — Dashboard UI/UX**

### 1.2. 목적 (Purpose)
- 강화학습의 진행 과정(승률·손실·탐험률)과 에이전트의 실제 플레이를 **한눈에 읽히게** 하는 계측기형 대시보드를 구축한다.
- 복잡한 학습 지표를 시각화해 "지금 잘 배우고 있는가/어디서 추측하는가"를 직관적으로 파악하게 돕는다.

### 1.3. 핵심 차별점 (Key Differentiators)
1. **계측기 감성(Instrument Panel)**: 데이터 밀도가 높고 안정적인 레이아웃. 수치는 모노스페이스로 정렬해 HTS/오실로스코프 같은 신뢰감을 준다.
2. **의미 중심 색(Semantic Color)**: 안전/위험/추측을 색으로 즉시 구분하되, 색맹 대비를 위해 색에만 의존하지 않고 보조 표식(아이콘·패턴)을 병행한다.
3. **절제(Anti-Cliché)**: 네온 글로우, 다중 색상 그라데이션, 그라데이션 텍스트, 글래스모피즘 남용, 떠다니는 blob 배경, 장식용 이모지 헤더를 **배제**하고 기능에 집중한다.

---

## 2. 상세 기능 요구사항 (Detailed Requirements)

### 2.1. 레이아웃 및 인터페이스 (Layout & Interface)
- **뷰 모드 (View Mode)**: Container-based, **Desktop First**. 좌측 고정 내비 + 메인 작업 영역 + 메트릭 패널의 다중 패널 구성.
  - *데스크톱*: 보드 뷰와 차트를 동시 표시하는 멀티 패널(예: 좌 내비 240px, 메인 가변, 우측/하단 메트릭).
  - *모바일*: 100% 단일 컬럼. 학습 모니터링 요약(승률·상태) 중심의 읽기 전용 축약.
- **테마 정책 (Theme Policy)**: 다크 우선 + 라이트 토글(System Preference 존중).
  - *배경색*: `--bg` #0e1116 (다크) / #f6f8fa (라이트).
  - *기본 텍스트 색*: `--text` #e6edf3 (다크) / #1f2328 (라이트).

### 2.2. 사용자 상호작용 (Interaction Logic)
- **주요 액션 (Actions)**:
  - **Hover Effects**: 보드 셀 호버 시 좌표·지뢰 확률 툴팁, 버튼은 배경색 1단계 변화 정도(과한 scale·rotate·glow 금지).
  - **Navigation**: 좌측 사이드바(GNB) — Train / Play / Benchmark / Models. Sticky 고정.
- **입력 방식 (Input)**: 학습 설정은 인라인 폼·슬라이더. 모달은 모델 가져오기/내보내기 등 한정 상황에만.

### 2.3. 데이터 구조 및 모듈 (Component Structure)
*대시보드 화면을 구성하는 주요 영역의 스타일 가이드.*
1. **헤더 (Header)**: 좌측 프로젝트명/로고. 우측 **서버 연결 상태 배지**(연결/끊김)와 **현재 학습 상태 배지**(running/paused/idle). 높이 56px, 하단 1px 보더.
2. **네비게이션 (Nav)**: 좌측 사이드바. Train/Play/Benchmark/Models 항목, 활성 항목 강조(좌측 4px 액센트 바). 배경 `--surface`, 그림자 없음(보더로 구분).
3. **콘텐츠 영역 (Content)**: 섹션별 패널 카드(`--surface`, 보더, 내부 패딩 16px). 보드 뷰·메트릭 차트·리포트 테이블이 배치됨. 강조는 색이 아니라 영역 구분과 여백으로.
4. **푸터 (Footer)**: 버전·문서 링크·페어플레이 주의 한 줄. 낮은 대비(`--text-muted`).

### 2.4. 출력 및 결과물 (Output)
- **결과물 형식**: React 컴포넌트(TSX) + CSS Variables 토큰.
- **품질 기준 (QA Standards)**:
  - 접근성: WCAG 2.1 AA 지향(대비비 확보, 키보드 내비, 의미 색 + 보조 표식 병행).
  - 반응형: 가로 스크롤 발생 금지. 좁은 화면에서 멀티 패널 → 단일 컬럼 스택.

---

## 3. 기술 스택 및 라이브러리 (Tech Stack)

### 3.1. Core
- **Frontend Framework**: React 18 + TypeScript + Vite.
- **Styling Engine**: **CSS Modules + CSS Variables**. 디자인 토큰을 변수로 단일 관리해 anti-cliché 일관성·다크/라이트 토글을 단순화. (Tailwind 대안 가능하나 토큰 명료성을 위해 CSS Variables 기준)

### 3.2. Libraries & Tools
1. **uPlot 또는 recharts**
   - **용도**: 학습 메트릭 차트. 고빈도 스트림은 uPlot, 정적 리포트는 recharts.
   - **설정 값**: 그라데이션 채움 비활성, 격자·축 라벨 명확, 라인 위주.
2. **lucide-react**
   - **용도**: 아이콘(상태 배지·내비). 장식이 아니라 의미 전달용으로 절제 사용.
3. **(선택) Radix UI primitives**
   - **용도**: 접근성 좋은 무스타일 컴포넌트(메뉴·다이얼로그). 확장성 확보.

---

## 4. 아키텍처 및 로직 (Architecture & Logic)

### 4.1. 시각적 계층 구조 (Visual Hierarchy)
- **Level 1 (Page/Section Title)**: 18px, Weight 600, `--text`.
- **Level 2 (Panel Title)**: 14px, Weight 600, `--text`, 하단 보더로 구분(밑줄 장식 금지).
- **Level 3 (Body)**: 14px, line-height 1.5, `--text`.
- **Level 4 (Meta/Caption)**: 12px, `--text-muted`.
- **수치(별도 트랙)**: 모노스페이스, tabular-nums로 자리 정렬. 큰 핵심 수치 20~24px.

```css
/* 스타일 적용 예시 */
.panel-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text);
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}
.metric-value {
  font-family: ui-monospace, "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums;
}
```

### 4.2. 반응형 로직 (Responsive Logic)
1. **Desktop (Default, >1024px)**: 멀티 패널(내비 + 보드 + 차트 동시).
2. **Transition Point**: 1024px / 640px.
3. **Mobile View (≤640px)**: 단일 컬럼 스택, 사이드바 → 상단 탭/햄버거, 차트 1단, 폰트 소폭 축소. 무거운 라이브 보드는 요약 카드로 대체 가능.

### 4.3. 핵심 컴포넌트 로직 (Core Components)
- **BoardView (가장 중요)**: 보드 셀 격자 렌더. 셀 상태(미오픈/숫자0~8/플래그/지뢰)와 **하이라이트 레이어**(확실 안전=녹 테두리, 확실 지뢰=적 표식, 추측=주의색 + 점선 같은 보조 패턴)를 분리해 그린다. 고전 베벨 느낌은 절제(과한 그림자·글로우 금지). 숫자 셀은 고전 지뢰찾기 색 차용.
- **MetricChart**: 승률 이동평균·손실·ε를 실시간 라인으로. 채움 그라데이션 금지, 축·범례 명확. 데이터가 고빈도라 다운샘플/스로틀.
- **RunControls / ConfigForm**: 학습 제어·설정. 슬라이더와 숫자 입력, 즉시 검증 피드백.
- **ModelList / BenchmarkTable**: 체크포인트 목록(에피소드·승률·포맷), 벤치 결과 테이블(모드·승률·평균 시간·3BV/s).

---

## 5. UI/UX 디자인 가이드 (Design System)

### 5.1. 색상 팔레트 (Color Palette)
- **Primary/Accent**: `#4c8dff` (주요 버튼·활성 강조 / `--accent`).
- **Secondary/Link**: `#58a6ff` (링크·보조 강조 / `--link`).
- **Background**: `#0e1116` (페이지 배경 / `--bg`), 표면 `#161b22` (`--surface`).
- **Text/Neutral**: `#e6edf3` (본문 / `--text`), `#8b949e` (보조 / `--text-muted`), 보더 `#2a313c` (`--border`).
- **Semantic**: 안전 `#2ea043` (`--safe`), 위험/지뢰 `#d2453f` (`--danger`), 추측/주의 `#d2a23f` (`--guess`).
- *라이트 테마는 동일 변수의 대체 값 세트로 매핑(배경 #f6f8fa, 텍스트 #1f2328 등).*

### 5.2. 타이포그래피 (Typography)
- **Font Family**: UI `system-ui, "Inter", sans-serif`. 수치 `ui-monospace, "JetBrains Mono", monospace`.
- **Font Weight**: Regular(400), Semibold(600), Bold(700). family 3종 이상 혼용 금지.

---

## 6. 파일 구조 (File Structure)
`@msai/web` 내부 디자인 관련 부분만 요약. **전체 트리는 별도 문서** `지뢰찾기AI_파일트리.md`.
```text
packages/web/src/
├── styles/
│   ├── global.css          # reset + 전역
│   └── tokens.css          # 디자인 토큰(CSS Variables, 다크/라이트)
├── components/
│   ├── layout/             # Sidebar, Header, Footer
│   ├── ui/                 # Button, Card, Badge, Modal (원자 단위)
│   └── features/           # BoardView, MetricChart, ConfigForm, ModelList, BenchmarkTable
└── views/                  # Train, Play, Benchmark, Models
```

---

## 7. 개발 시 주의사항 (Implementation Notes)
1. **스타일링 전략 (Styling Strategy)**:
   - 디자인 토큰(CSS Variables) 우선. 색·간격·반경 매직넘버 금지. CSS Modules로 스코프 격리, 클래스명은 BEM 또는 모듈 컨벤션.
2. **접근성 가이드 (Accessibility)**:
   - 아이콘·상태 배지에 `aria-label`, 키보드 내비 지원. 의미 색 + 보조 표식(아이콘/패턴) 병행으로 색맹 대응. 대비비 AA 확보.
3. **예외 처리 (Exception Handling)**:
   - 서버 미연결: 헤더 배지로 끊김 표시 + 재시도. 데이터 로딩: 스켈레톤 UI. 모델 없음: 플레이스홀더 카드. 학습 실패: 명확한 에러 패널(로그 링크).

---

> Anti-cliché 원칙 요약(반드시 준수): 네온/다중 그라데이션/그라데이션 텍스트/글래스모피즘 도배/blob 배경/장식 이모지 헤더/과한 호버 모션/폰트 3종+ 혼용 금지. "v0·Lovable에서 갓 뽑은 듯한" 신호를 제거하고 계측기형 절제미를 유지한다.
