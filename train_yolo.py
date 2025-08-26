from ultralytics import YOLO

if __name__ =="__main__":

    model = YOLO("yolo11n.pt")
    model.train(
        data = "datasets/yolo.yaml",
        imgsz = 800,
        epochs = 150,
        patience = 50,
        project = 'runs',
        name = 'yolo',
        # single_cls = True,
        # multi_scale = True,
        # cos_lr = True,
        # close_mosaic = 30,
        # dropout = 0.1,
        plots = True,
        degrees = 90,
        flipud = 0.5,
        fliplr = 0.5,
        mixup = 0.1,
        nms = False
    )
