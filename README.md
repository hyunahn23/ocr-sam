# ocr-sam
SegmentAnything 기반의 application인 OCR-SAM. 객체 분할을 활용한 application

환경설정
- Python 3.8
- PyTorch 2.0.1
- CUDA 11.8
- pip install -U openmim
- mim install mmengine
- mim install 'mmcv==2.0.0rc4'
- mim install 'mmcls==1.0.0rc5'
- mim install 'mmdet==3.0.0rc5'
- mim install mmocr
- pip install git+https://github.com/facebookresearch/segment-anything.git


Pretrained-Weight
# Text detection model
gdown 1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-

# mmocr recognizer
wget https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth

# segment anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


Config
- config 코드는 OCR-SAM repository 내 mmocr_dev/configs 폴더 참조


Code
- 코드의 base는 https://github.com/yeungchenwa/OCR-SAM 의 코드를 참고하여 mmocr과 SAM을 직접 활용하는 방안을 터득하고자 mmocr_sam코드를 기반으로 직접 구현.
