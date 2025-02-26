from ultralytics import YOLO
import time

def train():
    data_path = "hripcb.yaml"
    eps = 100
    img_size = 640
    project_name = "yolo11_hripcb_master"
    model_name = "yolo11n"
    batch_size = 16
    model = YOLO(model_name + ".pt")

    model.train(
        data=data_path,
        epochs=eps,
        imgsz=img_size,
        device=0,
        batch=batch_size,
        project=project_name,
        name=f"{model_name}_pcb_{eps}ep_{img_size}",
        exist_ok=True
    )

def val():
    model = YOLO("yolo11n.pt")

    validation_results = model.val(
        data="hripcb.yaml",
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=0
    )

def measure_inference_speed():
    model = YOLO("./yolo11_hripcb_master/yolo11n_pcb_100ep_640px/weights/best.pt")
    
    source = "./hripcb/test/images"

    results = model.predict(source, imgsz=640, device=0)
    total_fps = 0
    for result in results:
        fps = 1 / (result.speed["inference"] / 1000)
        total_fps += fps
    avg_fps = (total_fps / len(results))
    
    print(f"Avg FPS: {avg_fps:.2f}")

train()
val()
measure_inference_speed()
