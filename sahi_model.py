from sahi.sahi.auto_model import AutoDetectionModel
from sahi.sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from sahi.sahi.utils.cv import read_image


yolov8_model_path = "/Users/bhuvanrj/Desktop/cricket/ball/last_100.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.45,
    device="cpu", # or 'cuda:0'
)



result3 = get_sliced_prediction(
    "/Users/bhuvanrj/Desktop/cricket/ball/demo_data/ss.png",
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)
result3.export_visuals(export_dir="/Users/bhuvanrj/Desktop/cricket/ball/demo_data/")
for prediction in result3.object_prediction_list:
    # Extracting bbox values
    bbox = prediction.bbox
    x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
    width = bbox.maxx - bbox.minx
    height = bbox.maxy - bbox.miny

    # Extracting score value
    score = prediction.score.value

    # Extracting category values
    category_id = prediction.category.id
    category_name = prediction.category.name

    # Printing the values
    print(f"Bbox: ({x_min}, {y_min}, {x_max}, {y_max}), Width: {width}, Height: {height}, Score: {score}, Category: {category_name}")
    print(f"Bbox: ({x_min}, {y_min}, {x_max}, {y_max}), Score: {score}, Category: {category_name}")