# photo-restore-server

RTX 5070 Ti 16GB Linux 워크스테이션에서 동작하는 Docker 기반 사진 복원 API 서버입니다.

파이프라인은 다음 순서로 동작합니다.

1. 업로드된 이미지 저장
2. 너무 큰 입력은 작업용 해상도로 축소 (`MAX_INPUT_LONG_SIDE`)
3. 배경/전체 복원: SwinIR 또는 경량 보정 (`bg_model`)
4. 얼굴 복원: GFPGAN 또는 CodeFormer (`face_model`)
5. Real-ESRGAN 최종 업스케일
6. 출력 픽셀 수 자동 제한 (`MAX_OUTPUT_PIXELS`)
7. JPEG/WebP/PNG 저장 후 API 응답 반환

## 프로젝트 구조

```text
photo-restore-server/
├── docker-compose.yml
├── README.md
└── api/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py
    ├── worker.py
    ├── models/
    ├── input/
    ├── output/
    └── pipelines/
        ├── restore.py
        ├── upscale.py
        ├── codeformer.py
        ├── swinir.py
        ├── utils.py
        └── archs/
            ├── codeformer_arch.py
            └── vqgan_arch.py
```

## 지원 기능

- `POST /restore` 멀티파트 이미지 업로드
- `mode` 옵션 지원: `face`, `upscale`, `full`, `safe`, `pretty`, `restore`, `enhance`
- `upscale` 옵션 지원: `1`, `2`, `4`
- `fidelity` 옵션 지원: `0.0 ~ 1.0`
- `face_model` 옵션 지원: `auto`, `gfpgan`, `codeformer`
- `face_weight` 옵션 지원: `0.0 ~ 1.0` (`fidelity`보다 우선)
- `bg_model` 옵션 지원: `auto`, `swinir`, `light`, `none`
- 출력 포맷 지원: `jpg`, `webp`, `png` (`format`, `quality`)
- 업로드 크기 제한 지원: 기본 25MB (`MAX_UPLOAD_BYTES`)
- 입력 작업 해상도 제한: 기본 장변 3600px (`MAX_INPUT_LONG_SIDE`)
- 출력 픽셀 수 제한: 기본 40MP (`MAX_OUTPUT_PIXELS`)
- GPU 단일 워커 큐 기반 처리
- 결과 이미지 `api/output/` 저장
- 모델, 입력, 출력 디렉터리 영속 볼륨 유지

## 모델 파일 준비

컨테이너 실행 전 아래 필수 모델 파일을 `api/models/` 디렉터리에 넣어야 합니다.

- `GFPGANv1.4.pth`
- `RealESRGAN_x2.pth`
- `RealESRGAN_x4.pth`

아래 모델은 선택 사항이며, 파일이 있으면 해당 기능이 활성화됩니다.

- `CodeFormer.pth` (또는 `codeformer.pth`): CodeFormer 얼굴 복원.
  <https://github.com/sczhou/CodeFormer/releases> 에서 `codeformer.pth` 다운로드
- `SwinIR-M_x4_GAN.pth` (또는 `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-M_x4_GAN.pth`): SwinIR 전체 이미지 복원.
  <https://github.com/JingyunLiang/SwinIR/releases> 에서 다운로드

예시:

```bash
mkdir -p api/models
```

모델 파일명은 코드와 정확히 일치해야 합니다. CodeFormer/SwinIR 가중치가 없으면
`pretty`/`enhance` 모드는 기존 GFPGAN + 경량 보정으로 동작하고, `face_model=codeformer`
또는 `bg_model=swinir`를 명시적으로 요청한 경우에만 에러를 반환합니다.

## 실행 방법

```bash
docker compose up -d --build
```

## GPU 확인

```bash
docker exec -it photo-restore-api nvidia-smi
```

## 서버 상태 확인

```bash
curl http://localhost:8010/health
```

## 모델 준비 상태 확인

```bash
curl http://localhost:8010/ready
```

`/ready`는 필수 모델 파일이 모두 있는지 확인합니다. 필수 모델이 빠져 있으면 HTTP 503을 반환하며,
응답의 `optional_models` 필드에서 CodeFormer/SwinIR 가중치 존재 여부를 확인할 수 있습니다.

## 테스트

```bash
curl -X POST http://localhost:8010/restore \
-F "file=@test.jpg" \
-F "mode=enhance" \
-F "upscale=2" \
-F "fidelity=0.7" \
-F "format=jpg" \
-F "quality=92" \
--output restored.jpg
```

CodeFormer + SwinIR 강제 지정 예시:

```bash
curl -X POST http://localhost:8010/restore \
-F "file=@test.jpg" \
-F "mode=enhance" \
-F "face_model=codeformer" \
-F "face_weight=0.5" \
-F "bg_model=swinir" \
-F "upscale=2" \
--output restored.jpg
```

## API 사양

### `POST /restore`

#### Form Data

- `file`: 입력 이미지 파일
- `mode`: `face`, `upscale`, `full`, `safe`, `pretty`, `restore`, `enhance`
- `upscale`: `1`, `2`, `4`
- `fidelity`: `0.0 ~ 1.0`
- `face_model`: `auto`, `gfpgan`, `codeformer` (기본값: `auto`)
- `face_weight`: `0.0 ~ 1.0` (지정 시 `fidelity` 대신 얼굴 복원 가중치로 사용)
- `bg_model`: `auto`, `swinir`, `light`, `none` (기본값: `auto`)
- `format`: `jpg`, `jpeg`, `webp`, `png` (기본값: `jpg`)
- `quality`: `1 ~ 100` (기본값: `92`, `jpg`/`webp`에서 사용)

#### mode 동작

- `face`: 얼굴 복원만 수행 (입력 해상도 유지)
- `upscale`: Real-ESRGAN 업스케일만 수행
- `full`: 배경 보정 후 얼굴 복원, Real-ESRGAN 업스케일 수행
- `restore`: `full`과 동일한 현실적 이름의 alias
- `safe`: 낮은 강도의 배경/얼굴 복원 후 업스케일 수행
- `pretty`: 높은 강도의 배경/얼굴 복원 후 업스케일 수행
- `enhance`: `pretty`와 동일한 고강도 보정 alias

`face` 모드는 입력 해상도를 유지합니다. `full`, `safe`, `pretty`, `restore`, `enhance` 모드에서는 얼굴 복원 후 `upscale` 값이 최종 업스케일 배율로 적용됩니다.
기본 출력은 사진에 적합한 `jpg`이며, 무손실 결과가 필요할 때만 `format=png`를 사용하세요.

#### face_model 동작

- `auto`: `pretty`/`enhance` 모드에서 CodeFormer 가중치가 있으면 CodeFormer, 그 외에는 GFPGAN
- `gfpgan`: 항상 GFPGAN 사용
- `codeformer`: 항상 CodeFormer 사용 (가중치 없으면 에러)

CodeFormer의 `face_weight`는 낮을수록 강한 복원, 높을수록 원본 유지입니다.
`face_weight`를 지정하지 않으면 모드 preset에 따라 자동 결정됩니다
(CodeFormer 기준 `pretty`/`enhance`는 0.6 이하, `safe`는 0.8 이상).

#### bg_model 동작 (`full`/`safe`/`pretty`/`restore`/`enhance` 모드에서 적용)

- `auto`: `pretty`/`enhance` 모드에서 SwinIR 가중치가 있으면 SwinIR, 그 외에는 `light`
- `swinir`: SwinIR 전체 이미지 복원 (denoise, JPEG artifact 제거, 가중치 없으면 에러)
- `light`: CLAHE + denoise + sharpen 경량 보정 (기존 동작)
- `none`: 배경 처리 생략

#### 응답

- 성공 시 `format`에 해당하는 이미지 파일 반환 (`image/jpeg`, `image/webp`, `image/png`)
- 실패 시 JSON 에러 반환

## Docker 구성

- 서비스명: `photo-restore-api`
- 포트: `8010:8010`
- 베이스 이미지: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`
- PyTorch: CUDA 12.8 wheel 사용. RTX 50 시리즈는 `sm_120` 지원이 필요하므로 CUDA 12.4 wheel에서는 추론이 실패합니다.
- 볼륨:
  - `./api/models:/app/models`
  - `./api/input:/app/input`
  - `./api/output:/app/output`
- GPU 예약: NVIDIA runtime 기반 GPU 1개 사용
- Real-ESRGAN은 기본 `REAL_ESRGAN_TILE=512`로 타일 처리합니다. 큰 해상도 이미지가 CUDA OOM을 내면 값을 `256`으로 낮추고, 더 빠른 처리가 필요하고 VRAM 여유가 있으면 `768` 또는 `0`으로 조정할 수 있습니다.
- SwinIR은 기본 `SWINIR_TILE=256`, `SWINIR_TILE_OVERLAP=32`로 타일 처리합니다. OOM이 나면 `SWINIR_TILE`을 `128`로 낮추세요.
- `MAX_INPUT_LONG_SIDE=3600`: 장변이 이 값보다 큰 입력은 작업용으로 축소합니다. `0`이면 비활성화됩니다.
- `MAX_OUTPUT_PIXELS=40000000`: 최종 출력이 이 픽셀 수를 넘으면 자동 축소합니다. `0`이면 비활성화됩니다.

## 워커 구조

- API는 요청을 수신하고 파일을 저장합니다.
- 실제 GPU 추론은 `worker.py`의 단일 워커에서 수행합니다.
- 기본값은 `MAX_GPU_WORKERS=1` 이며 동시 GPU 실행을 막습니다.
- 기본 업로드 제한은 `MAX_UPLOAD_BYTES=26214400` 입니다.
- 기본 업스케일 타일 크기는 `REAL_ESRGAN_TILE=512` 입니다.
- 향후 Redis, Celery, RQ 같은 외부 큐로 확장할 수 있도록 분리돼 있습니다.

## 향후 확장 예정

현재 버전에는 포함하지 않았지만 다음 단계로 확장할 수 있습니다.

- SUPIR 기반 고품질 생성 복원
- Telegram Bot 연동
- openclaw HTTP API 연동

## 주의사항

- 이 서버는 GFPGAN + Real-ESRGAN을 기본으로 사용하고, 가중치가 있으면 CodeFormer + SwinIR을 추가로 사용합니다.
- SUPIR, Redis, Celery, DB는 현재 구현에 포함되지 않습니다.
- 모델 가중치 파일은 라이선스와 배포 정책에 맞게 직접 준비해야 합니다.
- CodeFormer 아키텍처 코드(`api/pipelines/archs/`)는 S-Lab License 1.0 (비상업용) 하에 배포되는 원본 저장소에서 가져온 것입니다. 상업적 사용 시 라이선스를 확인하세요.
