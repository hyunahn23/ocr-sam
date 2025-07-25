# OCR-SAM
**Segment Anything + MMOCR 기반의 OCR Application**

OCR-SAM은 Meta의 **Segment Anything** 모델과 OpenMMLab의 **MMOCR** 프레임워크를 결합하여, 객체 분할(SAM)을 통해 텍스트 인식 정확도를 높이고자 한 OCR 기반 응용 프로그램입니다.

---

## 환경 설정

- Python 3.8  
- PyTorch 2.0.1  
- CUDA 11.8  

필수 패키지 설치:

```bash
pip install -U openmim
mim install mmengine
mim install 'mmcv==2.0.0rc4'
mim install 'mmcls==1.0.0rc5'
mim install 'mmdet==3.0.0rc5'
mim install mmocr
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Pretrained Weights 다운로드
```bash
# Text detection model (gdown 사용)
gdown 1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-

# MMOCR recognizer (ABINet 모델)
wget https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth

# Segment Anything 모델 (ViT-B)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

##  Config 파일

config 코드는 OCR-SAM repository 내 mmocr_dev/configs 폴더 참조

---

## 코드 구조

- [yeungchenwa/OCR-SAM](https://github.com/yeungchenwa/OCR-SAM) 프로젝트를 참고하여 구현
- MMOCR과 Segment Anything(SAM)을 직접 연동하여 객체 분할 + 텍스트 인식을 결합
- 핵심 로직은 `mmocr_sam/` 디렉토리에 구현되어 있으며, SAM 마스크 기반으로 텍스트 인식 흐름을 처리.

---
