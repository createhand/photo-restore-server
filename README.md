# photo-restore-server

RTX 5070 Ti 16GB Linux 워크스테이션에서 동작하는 Docker 기반 사진 복원 API 서버입니다.

MVP 파이프라인은 다음 순서로 동작합니다.

1. 업로드된 이미지 저장
2. GFPGAN 얼굴 복원
3. Real-ESRGAN 업스케일
4. 결과 PNG 저장
5. API 응답 반환

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
        └── utils.py
```

## 지원 기능

- `POST /restore` 멀티파트 이미지 업로드
- `mode` 옵션 지원: `face`, `upscale`, `full`, `safe`, `pretty`
- `upscale` 옵션 지원: `1`, `2`, `4`
- `fidelity` 옵션 지원: `0.0 ~ 1.0`
- 업로드 크기 제한 지원: 기본 25MB (`MAX_UPLOAD_BYTES`)
- GPU 단일 워커 큐 기반 처리
- 결과 이미지 `api/output/` 저장
- 모델, 입력, 출력 디렉터리 영속 볼륨 유지

## 모델 파일 준비

컨테이너 실행 전 아래 모델 파일을 `api/models/` 디렉터리에 넣어야 합니다.

- `GFPGANv1.4.pth`
- `RealESRGAN_x2.pth`
- `RealESRGAN_x4.pth`

예시:

```bash
mkdir -p api/models
```

모델 파일명은 코드와 정확히 일치해야 합니다.

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

`/ready`는 필수 모델 파일이 모두 있는지 확인합니다. 모델이 빠져 있으면 HTTP 503을 반환합니다.

## 테스트

```bash
curl -X POST http://localhost:8010/restore \
-F "file=@test.jpg" \
-F "mode=full" \
-F "upscale=2" \
-F "fidelity=0.7" \
--output restored.png
```

## API 사양

### `POST /restore`

#### Form Data

- `file`: 입력 이미지 파일
- `mode`: `face`, `upscale`, `full`, `safe`, `pretty`
- `upscale`: `1`, `2`, `4`
- `fidelity`: `0.0 ~ 1.0`

#### mode 동작

- `face`: GFPGAN 얼굴 복원만 수행
- `upscale`: Real-ESRGAN 업스케일만 수행
- `full`: GFPGAN 후 Real-ESRGAN 수행
- `safe`: 낮은 강도의 얼굴 복원 후 업스케일 수행
- `pretty`: 높은 강도의 얼굴 복원 후 업스케일 수행

`face` 모드는 입력 해상도를 유지합니다. `full`, `safe`, `pretty` 모드에서는 얼굴 복원 후 `upscale` 값이 최종 업스케일 배율로 적용됩니다.

#### 응답

- 성공 시 `image/png` 파일 반환
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

## 워커 구조

- API는 요청을 수신하고 파일을 저장합니다.
- 실제 GPU 추론은 `worker.py`의 단일 워커에서 수행합니다.
- 기본값은 `MAX_GPU_WORKERS=1` 이며 동시 GPU 실행을 막습니다.
- 기본 업로드 제한은 `MAX_UPLOAD_BYTES=26214400` 입니다.
- 기본 업스케일 타일 크기는 `REAL_ESRGAN_TILE=512` 입니다.
- 향후 Redis, Celery, RQ 같은 외부 큐로 확장할 수 있도록 분리돼 있습니다.

## 향후 확장 예정

현재 버전에는 포함하지 않았지만 다음 단계로 확장할 수 있습니다.

- CodeFormer 기반 `pretty` 전용 경로 고도화
- SUPIR 기반 고품질 생성 복원
- Telegram Bot 연동
- openclaw HTTP API 연동

## 주의사항

- 이 MVP는 GFPGAN + Real-ESRGAN만 사용합니다.
- CodeFormer, SUPIR, Redis, Celery, DB는 현재 구현에 포함되지 않습니다.
- 모델 가중치 파일은 라이선스와 배포 정책에 맞게 직접 준비해야 합니다.
