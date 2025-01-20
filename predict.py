from ultralytics import YOLO

# Caminho para o modelo salvo
model_path = "./runs/classify/train4/weights/best.pt"

# Carregar o modelo
model = YOLO(model_path)
# Caminho para imagem de validação
val_image_path = "./path/to/image.jpg"
# Usar o modelo para inferência
results = model(val_image_path)
results.show()
