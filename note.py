import os
import shutil
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO

history = None
def train_yolo_model(data_dir, epochs=50, imgsz=224, batch_size=16):
    # Caminho para o modelo pré-treinado YOLOv8
    model = YOLO('yolov8n-cls.pt')  # Ou qualquer outro modelo pré-treinado que você queira usar
    print("asdsadasdasd")
    # Treinamento do modelo YOLOv8
    results = model.train(
        data=data_dir,  # Caminho para o diretório com a estrutura de treino, val e test
        epochs=epochs,  # Número de épocas
        imgsz=imgsz,     # Tamanho da imagem para treinamento
        batch=batch_size # Tamanho do lote
    )
    history = results.metrics
    # Mostrar os resultados do treinamento
    print("Treinamento finalizado!")
    print(f"Resultados do treino: {results}")
    
    # Exibir a evolução das métricas durante o treinamento
    plot_training_curves(results.metrics)

    # Avaliar o modelo com a matriz de confusão
    plot_confusion_matrix(model, data_dir)
    
    return model

def plot_training_curves(history):
    # Função para plotar as curvas de treinamento e validação
    import matplotlib.pyplot as plt
    
    # Plotando a perda (loss) de treinamento e validação
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plotando a acurácia de treinamento e validação
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(model, data_dir):
    # Fazer previsões no conjunto de teste
    results = model.predict(source=os.path.join(data_dir, 'test'))  # Passa o caminho do diretório de teste

    # Obter as previsões e as etiquetas reais
    preds = results.pred
    true_labels = results.labels

    # Gerar a matriz de confusão
    cm = confusion_matrix(true_labels, preds)

    # Plotar a matriz de confusão usando seaborn para uma melhor visualização
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.names, yticklabels=model.names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

yolo_dataset_path = "./organized_dataset"
model = train_yolo_model(yolo_dataset_path)
print(model)
plot_training_curves(history)
plot_confusion_matrix(model, yolo_dataset_path)