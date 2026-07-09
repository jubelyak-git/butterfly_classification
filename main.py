import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

WEIGHTS_PATH = "model_weightss_butterfly2.tar"
VECTOR_DB_PATH = "my_vector_db.pt"

BUTTERFLIES = [
    "Argynnis_hyperbius",
    "Cephonodes_hylas",
    "Eurema_mandarina",
    "Graphium_sarpedon_nipponum",
    "Hestina_assimilis_assimilis",
    "Lampides_boeticus",
    "Papilio_xuthus",
    "Polygonia_c-aureum",
    "Pseudozizeeria_maha",
    "Сyrestis_thyodamas"
]

model = None
device = None
vector_db = None
# Обрабатываем фотографию
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(weights_file=WEIGHTS_PATH):
    global model, device
    if model is not None:
        return model

    device = torch.device("cpu")
    #загружаем пустую реснет 50 / сохраняем размер последнего слоя
    net = models.resnet50(weights=None)
    in_feat = net.fc.in_features

    weights = torch.load(weights_file, map_location="cpu")

    if isinstance(weights, dict):
        state = weights.get("model_state_dict", weights.get("state_dict", weights))
    else:
        raise RuntimeError("Invalid weights file")

    cleaned_state = {k.replace("module.", ""): v for k, v in state.items()}
    # наверное можно убрать иф динамические
    if "fc.weight" in cleaned_state:
        num_classes = cleaned_state["fc.weight"].shape[0]
        net.fc = nn.Linear(in_feat, num_classes)
    else:
        net.fc = nn.Linear(in_feat, len(BUTTERFLIES))

    net.load_state_dict(cleaned_state, strict=False)
    net.to(device)
    net.eval()

    model = net
    return model


def load_vector_db():
    global vector_db
    if vector_db is not None:
        return vector_db

    if not os.path.exists(VECTOR_DB_PATH):
        print(f"ВНИМАНИЕ: Файл базы {VECTOR_DB_PATH} не найден. Сначала запусти create_db.py")
        return None

    vector_db = torch.load(VECTOR_DB_PATH, map_location="cpu")
    print("Векторная база данных загружена.")
    return vector_db

# класфификаци бабочек
def predict_butterfly(image_path):
    net = load_model()
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    species = BUTTERFLIES[pred_idx]
    print(f"Классификация: {species} ({confidence:.2%})")
    return species

# Принимает путь к картинке, генерирует эмбеддинг
#     и возвращает пути к 3 самым похожим картинкам из базы
def get_similar_images(image_path, top_k=3):

    net = load_model() # наша модель
    db = load_vector_db() # векторная база

    if db is None:
        return []

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
#извлечение эмбэдингов
    with torch.no_grad():
        original_fc = net.fc
        net.fc = nn.Identity()

        query_embedding = net(tensor).cpu()

        net.fc = original_fc

    db_embeddings = db["embeddings"]
    db_paths = db["paths"]
# норализируем векторы
    query_norm = F.normalize(query_embedding, p=2, dim=1)
    db_norm = F.normalize(db_embeddings, p=2, dim=1)
# вычисления сходства
    similarities = torch.mm(query_norm, db_norm.t()).squeeze(0)

    scores, indices = torch.topk(similarities, k=top_k)

    results = []
    for i in range(top_k):
        idx = indices[i].item()
        score = scores[i].item()
        results.append(db_paths[idx])
        print(f"Похожее фото {i + 1}: {db_paths[idx]} (сходство: {score:.4f})")

    return results


def show_results(query_path, similar_paths):
    """
    Отображает исходное изображение и похожие на него.
    """
    n = len(similar_paths)
    plt.figure(figsize=(16, 5))

    plt.subplot(1, n + 1, 1)
    query_img = Image.open(query_path).convert("RGB")
    plt.imshow(query_img)
    plt.axis('off')

    for i, path in enumerate(similar_paths):
        plt.subplot(1, n + 1, i + 2)
        try:
            sim_img = Image.open(path).convert("RGB")
            plt.imshow(sim_img)
            plt.title(f"Похожее #{i + 1}")
        except Exception as e:
            plt.text(0.5, 0.5, f"Ошибка загрузки\n{os.path.basename(path)}",
                     ha='center', va='center')
            print(f"Не удалось открыть файл: {path}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_image = "image copy 3.png"
    species = predict_butterfly(test_image)
    print(f"Это бабочка вида: {species}")
    similar_images = get_similar_images(test_image, top_k=3)
    show_results(test_image, similar_images)