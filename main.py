import cv2
import numpy as np
import torch
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox
from segment_anything import SamPredictor, sam_model_registry


def visualize(input_img, mask):
    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 3), dtype=np.uint8)
    squeezed_mask = mask.squeeze()

    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

    mask_image[squeezed_mask] = color
    alpha = 0.5
    overlay_mask = cv2.addWeighted(input_img, 1 - alpha, mask_image, alpha, 0)

    return overlay_mask


if __name__=='__main__':
    inputs_dir = "inputs/imgs/"
    outputs_dir = "results/"

    text_det_cfg_path = 'configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k.py'

    text_det_ckpt_path = 'weights/db_swin_mix_pretrain.pth'
    text_rec_cfg_path = 'configs/textrecog/abinet/abinet_20e_st-an_mj.py'
    text_rec_ckpt_path = 'weights/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth'
    sam_model_type = 'vit_h'
    sam_ckpt_path = 'weights/sam_vit_h_4b8939.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    ocr_inferencer = MMOCRInferencer(
        text_det_cfg_path,
        text_det_ckpt_path,
        text_rec_cfg_path,
        text_rec_ckpt_path,
        device=device)
    
    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_ckpt_path)
    sam_model.to(device=device)
    sam_predictor = SamPredictor(sam_model)

    original_inputs = ocr_inferencer._inputs_to_list(inputs_dir)

    for _, original_input_path in enumerate(original_inputs):
        input_img = cv2.imread(original_input_path)

        ocr_result = ocr_inferencer(input_img, save_vis=True, out_dir=outputs_dir)['predictions'][0]

        recognized_texts = ocr_result['rec_texts']
        detected_polygons = ocr_result['det_polygons']

        detected_bboxes = torch.tensor(np.array([poly2bbox(polygon) for polygon in detected_polygons]), device=device)
        transformed_bboxes = sam_predictor.transform.apply_boxes_torch(detected_bboxes, input_img.shape[:2])

        sam_predictor.set_image(input_img, image_format='BGR')

        masks, scores, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_bboxes,
            multimask_output=False)
        
        for mask in masks:
            mask_array = mask.to('cpu').numpy()
            visualize(input_img, mask_array)

        output_path = outputs_dir + original_input_path
        cv2.imwrite(output_path, input_img)