from ultralytics import YOLO

if __name__ == "__main__":
    # 1) Carrega peso pré-treinado (faz transfer learning)
    model = YOLO('yolov8n.pt')

    # 2) Treinamento customizado
    results = model.train(
        data='dataset/data.yaml',
        epochs=80,            # epochs moderadas
        batch=8,              # batch pequeno para 100 imgs
        imgsz=640,
        name='thermal_small',
        device='cpu',        # ou 'cpu' se não tiver GPU
        workers=4,
        lr0=0.001,            # learning rate baixo para evitar overfitting
        weight_decay=0.0001,  # regularização extra
        patience=10,          # early stopping
        augment=True,         # ativa mosaic, mixup e flips
        freeze=10             # congela backbone nas 10 primeiras épocas
    )

    # 3) Exibe resumo dos resultados
    print(results)
