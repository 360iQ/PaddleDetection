import json
from PIL import Image
import os
import shutil
from collections import defaultdict
import argparse

SCORE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.8

parser = argparse.ArgumentParser(description='Customize Paths')
parser.add_argument('--image_dir', type=str, default='/opt/ml/input/dataset/custom',
                    help='Path to the directory containing images')
parser.add_argument('--model_names', nargs='+',
                    default=['ppyoloe_plus_crn_l_80e_custom', 'rtdetr_r101vd_6x_custom'],
                    help='List of model names')
parser.add_argument('--output_dir_labelled', type=str, default='/opt/ml/output/labelled_images',
                    help='Path to the output directory')

args = parser.parse_args()


def get_image_dimensions(image_dir):
    image_dimensions = {}
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            with Image.open(os.path.join(image_dir, image_name)) as img:
                width, height = img.size
            image_dimensions[image_name] = (width, height)
    return image_dimensions

def convert_to_coco_format(old_bbox_data, image_dimensions):
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "car", "supercategory": ""}]
    }
    
    filename_to_img_id = {filename: img_id for img_id, (filename, _) in enumerate(image_dimensions.items(), start=1)}
    annotation_id = 1
    
    for img_id, (image_file_name, (width, height)) in enumerate(image_dimensions.items(), start=1):
        coco_format["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": image_file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        
    for annotation in old_bbox_data:
        if annotation.get("score", 0) < SCORE_THRESHOLD:
            continue

        image_id = filename_to_img_id.get(annotation['file_name'], None)
        if image_id is None:
            continue

        # area = annotation["bbox"][2] * annotation["bbox"][3]
        x, y, width, height = annotation["bbox"]
        
        # Calculate area
        area = width * height

        # Filter out bounding boxes based on dimensions and area
        if (width < 25 and height < 25) or area < 600:
            continue  
        
        
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [],
            "area": area,
            "bbox": annotation["bbox"],
            "iscrowd": 0
        })
        
        annotation_id += 1
    
    return coco_format


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def copy_and_update_list(image, file_name, destination, img_list, image_dir, annotations=None):
    shutil.copy(os.path.join(image_dir, file_name), destination)
    img_list["images"].append(image)
    if annotations is not None:
        img_list["annotations"].extend(annotations)

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1, xi2, yi2 = max(x1, x2), max(y1, y2), min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area, box2_area = w1 * h1, w2 * h2
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def main():
    image_dir = args.image_dir  # Updated
    image_dimensions = get_image_dimensions(image_dir)
    # model_names = args.model_names
    output_dir = os.path.join(args.output_dir_labelled, 'compared') 
    
    os.makedirs(f"{output_dir}/Agree", exist_ok=True)
    os.makedirs(f"{output_dir}/Disagree", exist_ok=True)
    os.makedirs(f"{output_dir}/Background", exist_ok=True)

    # Initialize categorized lists
    agree_list = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": [{"id": 1, "name": "car", "supercategory": ""}]}
    disagree_list = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": [{"id": 1, "name": "car", "supercategory": ""}]}
    background_list = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": [{"id": 1, "name": "car", "supercategory": ""}]}
    
    for model_name in args.model_names:
        bbox_path = os.path.join(args.output_dir_labelled, model_name, "bbox.json")
        with open(bbox_path, 'r') as f:
            old_bbox_data = json.load(f)

        coco_format_data = convert_to_coco_format(old_bbox_data, image_dimensions)
        
        save_json(coco_format_data, os.path.join(args.output_dir_labelled, model_name, "bbox_cvat.json"))

    model1_bbox_path = os.path.join(args.output_dir_labelled, args.model_names[0], 'bbox_cvat.json')
    model2_bbox_path = os.path.join(args.output_dir_labelled, args.model_names[1], 'bbox_cvat.json')

    with open(model1_bbox_path, 'r') as f1, open(model2_bbox_path, 'r') as f2:
        model1_data, model2_data = json.load(f1), json.load(f2)

    annotations1, annotations2 = defaultdict(list), defaultdict(list)

    for ann in model1_data["annotations"]:
        annotations1[ann["image_id"]].append(ann)
    for ann in model2_data["annotations"]:
        annotations2[ann["image_id"]].append(ann)

    for image in model1_data["images"]:
        img_id = image["id"]
        file_name = image["file_name"]
        boxes1, boxes2 = annotations1.get(img_id, []), annotations2.get(img_id, [])

        # if not boxes1 and not boxes2:
        #     copy_and_update_list(image, file_name, f"{output_dir}/Background", background_list)
        #     continue

        # if len(boxes1) != len(boxes2):
        #     copy_and_update_list(image, file_name, f"{output_dir}/Disagree", disagree_list, boxes2)
        #     continue

        # ious = [calculate_iou(b1["bbox"], b2["bbox"]) for b1, b2 in zip(boxes1, boxes2)]

        # if all(iou > IOU_THRESHOLD for iou in ious):
        #     copy_and_update_list(image, file_name, f"{output_dir}/Agree", agree_list, boxes1)
        # else:
        #     copy_and_update_list(image, file_name, f"{output_dir}/Disagree", disagree_list, boxes2)
        if not boxes1 and not boxes2:
            copy_and_update_list(image, file_name, f"{output_dir}/Background", background_list, image_dir)
            continue

        if len(boxes1) != len(boxes2):
            copy_and_update_list(image, file_name, f"{output_dir}/Disagree", disagree_list, image_dir, boxes2)
            continue

        ious = [calculate_iou(b1["bbox"], b2["bbox"]) for b1, b2 in zip(boxes1, boxes2)]

        if all(iou > IOU_THRESHOLD for iou in ious):
            copy_and_update_list(image, file_name, f"{output_dir}/Agree", agree_list, image_dir, boxes1)
        else:
            copy_and_update_list(image, file_name, f"{output_dir}/Disagree", disagree_list, image_dir, boxes2)


    save_json(agree_list, f"{output_dir}/agree_bbox.json")
    save_json(disagree_list, f"{output_dir}/disagree_bbox.json")
    save_json(background_list, f"{output_dir}/background_bbox.json")

        
if __name__ == '__main__':
    main()