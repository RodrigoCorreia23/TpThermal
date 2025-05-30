from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')

    # Treino
    results = model.train(
        data='dataset/data.yaml',
        epochs=80,            # epochs moderadas
        batch=8,              
        imgsz=640,
        name='thermal_small',
        device='cpu',        
        workers=4,
        lr0=0.001,            # learning rate baixo para evitar overfitting
        weight_decay=0.0001,  # regularização extra
        patience=10,          # early stopping
        augment=True,         # mosaic, mixup e flips
        freeze=10             # congela backbone nas 10 primeiras épocas
    )

    print(results)
