# 진행 상황 (2026-06-11)

- [x] 1. JPEG/WebP 출력 옵션 (`format`, `quality`) + `MAX_OUTPUT_PIXELS` 출력 제한
- [x] 2. 모드 정리 (`restore`/`enhance` alias, `safe`/`pretty`는 내부 preset으로 유지)
- [x] 3. CodeFormer 얼굴 복원 (`face_model=auto|gfpgan|codeformer`, `face_weight`, pretty/enhance preset이 CodeFormer 우선 사용)
- [x] 4. SwinIR 전체 이미지 복원 (`bg_model=auto|swinir|light|none`, 타일 추론, 1x 복원 후 Real-ESRGAN 최종 업스케일)
- [x] 5-보조. 큰 입력 작업용 축소 (`MAX_INPUT_LONG_SIDE=3600`)
- [ ] 5. SUPIR 고품질 생성 복원 (별도 실험, 16GB VRAM 튜닝 필요)

CodeFormer는 `CodeFormer.pth`, SwinIR은 `SwinIR-M_x4_GAN.pth` 가중치를 `api/models/`에 넣으면 활성화됩니다. 자세한 사용법은 README 참고.

---

아래는 원본 계획 메모.

품질을 올리려면 방향을 분리해야 합니다. 지금처럼 GFPGAN + Real-ESRGAN만으로는 “얼굴은 좋아지는데 전체 사진이 복원되는 느낌”까지 가기 어렵습니다.

제 판단으로는 순서가 이렇습니다.

1. 먼저 현재 API 결과물을 정상 사진 포맷으로 바꾸기

품질 개선은 아니지만 체감에 바로 영향 있습니다.

• 기본 출력: PNG → JPEG/WebP
• format=jpg|webp|png
• quality=90~95
• 너무 큰 출력은 자동으로 max_output_pixels 제한

이걸 안 하면 결과가 50MB씩 나오고, 클라이언트에서 보기/전송/저장이 다 불편합니다.

2. 모드 이름을 현실적으로 바꾸기

현재 safe, pretty는 이름만 그럴듯하고 실제로는 GFPGAN fidelity 차이입니다. 품질 차이가 작을 수밖에 없습니다.

지금은 이렇게 바꾸는 게 낫습니다:

• face: 얼굴만 복원
• upscale: 전체 업스케일
• restore: 얼굴 복원 + 업스케일
• enhance: 얼굴 복원 + 배경 업스케일 + JPEG/WebP 출력

safe/pretty는 지금 단계에서는 빼거나 내부 preset 정도로 낮추는 게 맞습니다.

3. 얼굴 품질은 CodeFormer 추가

GFPGAN은 얼굴을 예쁘게 복원하지만 가끔 너무 매끈하고 비슷비슷한 얼굴이 됩니다. CodeFormer는 blind face restoration용 모델이고, fidelity/quality 조절이 더 명확해서 “복원 강도” 차이를 만들기 좋습니다.

추천:

• face_model=gfpgan|codeformer
• face_weight=0.3~0.9
• pretty는 CodeFormer 강한 복원 preset으로 재정의

4. 배경/전체 사진 품질은 SwinIR 또는 SUPIR 계열 추가

배경이 거의 안 좋아지는 건 Real-ESRGAN이 “업스케일러”에 가깝기 때문입니다. 오래된 사진의 노이즈, JPEG artifact, 흐림, 손상 복원은 별도 image restoration 모델이 필요합니다.

현실적인 선택지는 둘입니다.

• SwinIR: denoise, JPEG artifact removal, super-resolution 쪽에 강함. API 서버에 붙이기 비교적 현실적.
• SUPIR: 훨씬 “AI가 새로 복원한 느낌”이 강함. 품질은 좋지만 무겁고, 16GB VRAM에서는 튜닝/타일링/저해상도 사전 리사이즈가 필요할 가능성이 큼.

5. 진짜 고품질 파이프라인은 2단계로 가야 함

추천 파이프라인:

입력 이미지
→ EXIF/해상도 확인
→ 필요 시 너무 큰 이미지는 장변 3000~4000px로 작업용 축소
→ 얼굴 검출/복원: GFPGAN 또는 CodeFormer
→ 전체 이미지 복원: SwinIR/Restormer/SUPIR 계열
→ 최종 업스케일: Real-ESRGAN x2
→ 출력: JPEG/WebP quality 92

MVP에서 바로 할 현실적인 다음 작업은 이겁니다:

1. JPEG/WebP 출력 옵션 추가
2. safe/pretty 모드 정리
3. CodeFormer 얼굴 복원 추가
4. SwinIR 배경/전체 복원 추가
5. SUPIR는 고품질 옵션으로 별도 실험

내 의견으로는 바로 SUPIR부터 붙이면 무겁고 불안정해질 확률이 높고, 먼저 JPEG/WebP 출력 + CodeFormer + SwinIR 순서가 좋습니다. 이러면 API 안정성 유지하면서 “모드별 결과 차이”도 확실히 만들 수 있습니다.

참고한 공식/원본 프로젝트:

• GFPGAN: https://github.com/TencentARC/GFPGAN
• Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
• CodeFormer: https://github.com/sczhou/CodeFormer
• SwinIR: https://github.com/JingyunLiang/SwinIR
• SUPIR: https://github.com/Fanghua-Yu/SUPIR
