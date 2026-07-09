import streamlit as st
import os
import sys
from PIL import Image

# Добавляем родительскую папку в sys.path, чтобы видеть main.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

try:
    from main import load_model, predict_butterfly, get_similar_images
except ImportError as e:
    st.error(f"Ошибка импорта: {e}. Убедитесь, что вы запускаете streamlit из корневой папки проекта.")
    st.stop()

st.title('Классификация топ 10 бабочек из Японии')

try:
    load_model()
except Exception as e:
    st.warning(f"Не удалось загрузить веса модели: {e}")

uploaded_file = st.file_uploader(label='Загрузите файл для распознавания', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ваше изображение', width=300)

    temp_filename = "temp_query_image.jpg"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button('Узнать вид бабочки'):
        st.write("Обработка...")

        try:
            species_result = predict_butterfly(temp_filename)

            st.success(f"Результат классификации: **{species_result}**")

            st.markdown("### Похожие изображения из базы:")
            similar_paths = get_similar_images(temp_filename, top_k=3)

            if similar_paths:
                cols = st.columns(3)
                for i, path in enumerate(similar_paths):
                    with cols[i]:
                        full_path = os.path.join(parent_dir, path) if not os.path.isabs(path) else path

                        if os.path.exists(full_path):
                            sim_img = Image.open(full_path)
                            st.image(sim_img, caption=f"Топ #{i + 1}")
                            st.caption(f"Файл: {os.path.basename(path)}")
                        else:
                            if os.path.exists(path):
                                st.image(Image.open(path), caption=f"Топ #{i + 1}")
                            else:
                                st.error(f"Файл не найден: {path}")
            else:
                st.info("Похожие изображения не найдены (возможно, база векторов пуста).")

        except Exception as e:
            st.error(f"Произошла ошибка при обработке: {e}")

