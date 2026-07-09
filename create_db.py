import os
import zipfile
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict


def main():
    zip_filename = "images_splited.zip"
    extract_to = "./data_extracted"
    data_root = os.path.join(extract_to, "images_splited", "train")

    output_filename = "my_vector_db.pt"
    batch_size = 32
    workers = 2
    SAMPLES_PER_CLASS = 30 # беру 30 картинок на класс чтобы быстрее работала

    if not os.path.exists(data_root):
        print(f"Распаковка {zip_filename}...")
        if not os.path.exists(zip_filename):
            print(f"ОШИБКА: Файл {zip_filename} не найден.")
            return
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"Данные найдены в {data_root}.")

    device = torch.device("cpu")
#обработка изображения
    db_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# инициализируем
    full_dataset = datasets.ImageFolder(root=data_root, transform=db_transform)

    indices_to_keep = []
    class_counts = defaultdict(int)
# проходи по всем классам берем первые 30
    for idx, (_, class_idx) in enumerate(full_dataset.samples):
        if class_counts[class_idx] < SAMPLES_PER_CLASS:
            indices_to_keep.append(idx)
            class_counts[class_idx] += 1
# создаем под датасет
    subset_dataset = Subset(full_dataset, indices_to_keep)
# делим данные на батчи
    db_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    print(f"Классов: {len(full_dataset.classes)}")
    print(f"Итого изображений для базы (по {SAMPLES_PER_CLASS} на класс): {len(subset_dataset)}")


    # только resnet без классификации  возвращает эмбеддинг размерности 2048
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity() # заменям
    backbone.to(device)
    backbone.eval()

    database = []
    print("Генерация эмбеддингов...")

    with torch.no_grad():
        for images, _ in tqdm(db_loader):
            images = images.to(device)
            features = backbone(images)
            database.append(features.cpu())

    if len(database) > 0:
        all_embeddings = torch.cat(database)
#gпути берутся в том же порядке
        all_paths = [full_dataset.samples[i][0] for i in indices_to_keep]

        db_payload = {
            "embeddings": all_embeddings,
            "paths": all_paths,
            "classes": full_dataset.classes
        }

        torch.save(db_payload, output_filename)
        print(f" База сохранена: {output_filename}")
    else:
        print("Ошибка: база пуста.")


if __name__ == '__main__':
    main()