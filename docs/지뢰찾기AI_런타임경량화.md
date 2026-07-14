# 지뢰찾기 AI 런타임 경량화

**적용일**: 2026-07-14

## 현재 상태

CUDA 학습 런타임은 유지하면서 재설치 가능한 파일과 현재 프로젝트에서 사용하지 않는
PyTorch 개발 파일을 제거했다.

| 단계 | 프로젝트 용량 | 절감량(최초 대비) |
| --- | ---: | ---: |
| 정리 전 | 4.789 GiB | - |
| 안전 정리 후 | 4.518 GiB | 277.5 MiB |
| 런타임 경량화 후 | 3.740 GiB | 1,074.3 MiB |

## 제거한 항목

- 사용하지 않는 Python Playwright 패키지.
- 재설치 가능한 루트 `node_modules`.
- Python, pytest, ruff 캐시.
- 실패한 M4-R2 v1-v3 사전학습 체크포인트와 manifest.
- `torch/**/*.lib` 22개, 769.5 MiB.
- `torch/include` 8,690개, 27.2 MiB.

최신 `axial-pretrained.pt`, 모든 checkpoint manifest, legacy 기준선, lifetime 통계,
CUDA DLL과 Python 모듈은 보존했다.

## 검증

- PyTorch 2.6.0+cu124, `torch.cuda.is_available() == True`.
- 최신 axial checkpoint load.
- RTX 4060 CUDA forward, backward, AdamW step.
- production 128x8 bootstrap 기반 masked PPO update와 checkpoint/manifest save.
- Python 테스트 44/44.
- ruff 전체 통과.
- dashboard HTTP/WebSocket 테스트 통과.

## 제한

현재 런타임은 일반 PyTorch 학습과 추론에 사용 가능하지만 다음 기능은 지원 대상으로
간주하지 않는다.

- `torch.utils.cpp_extension`.
- 사용자 정의 C++/CUDA extension compile/link.
- PyTorch header와 import library를 요구하는 외부 native build.
- 일부 `torch.compile` native backend.

프로젝트는 현재 위 기능을 사용하지 않는다. PyTorch 재설치 또는 업그레이드는 제거한
개발 파일을 다시 생성하므로 이후 용량을 재확인한다.

## 원복

개발 파일이 필요해지면 서버 디렉터리에서 동일 CUDA wheel을 재설치한다.

```powershell
uv pip install --python .venv\Scripts\python.exe --reinstall `
  "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

TypeScript 작업이 필요하면 루트에서 의존성을 복구한다.

```powershell
pnpm install
```

Playwright는 실사이트 자동화 폐기 결정에 따라 복구하지 않는다.
