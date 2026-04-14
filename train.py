from ultralytics import YOLO


def train_for_sdr_net():
    model = YOLO(
        r'C:\Users\29383\Desktop\ultralytics-main\ultralytics\cfg\models\11\yolo11-obb.yaml'
    ).load(
        r'C:\Users\29383\Desktop\ultralytics-main\my\yolo11n-obb.pt'
    )

    return model.train(
        data=r'C:\Users\29383\Desktop\ultralytics-main\my\data.yaml',
        device=0,
        imgsz=800,
        epochs=180,
        batch=8,
        warmup_epochs=10.0,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        degrees=180.0,
        mosaic=1.0,
        mixup=0.3,
        scale=0.6,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        close_mosaic=10,

        # SDR-Net custom args
        lambda_r=12.44,
        lambda_s=0.001,
        lambda_th=0.5,
        lambda_o=2.0,
        lambda_off=1.0,
        lambda_polar=0.5,
        beta_smooth_l1=0.11,

        amp=True,
        workers=8,
        val=True,
        plots=True,
        save=True,
        project='runs/train',
        name='RSDD-IN-SDR'
    )


if __name__ == '__main__':
    train_for_sdr_net()