import cv2
import numpy as np
from PIL import Image
import pandas as pd
import app
import pytesseract
import subprocess
import easyocr
import re
import os
import shutil

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
reader = easyocr.Reader(['ru'])


def remove_images():
    """
        Убирание прошлых изображений
    """
    shutil.rmtree(app.IMAGE_DIR)
    os.makedirs(app.IMAGE_DIR)


def crop_img(file):
    """
    Обрезание изображения
    :param file: загруженное пользователем изображение
    """
    # Считывание изображения
    img = Image.open(file.file)
    img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Присвоение изображению порогового значения в виде двоичного изображения
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Обнаружение контуров
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Определение самого большого контура
    largest_contour = max(contours, key=cv2.contourArea)

    # Обрезание по этому контуру
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = img[y:y + h, x:x + w]

    # Сохранение
    cv2.imwrite(app.cropped_path, cropped_image)


def acc_img():
    """
    Увеличение точности изображения
    """
    command1 = f'python inference_realesrgan.py --model_path RealESRGAN_x4plus.pth --input {app.cropped_path} '
    command2 = f'--output {app.IMAGE_DIR} --fp32'
    full_command = command1 + command2
    subprocess.run(full_command, shell=True)


def img2table(img_path):
    """
    Распознание таблицы с изображения
    :param img_path: путь до изображения
    """
    # Считывание изображения
    img = Image.open(img_path)
    img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Присвоение изображению порогового значения в виде двоичного изображения
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Ширина ядра как 100-я часть общей ширины
    kernel_len = np.array(img).shape[1] // 100

    # Определение вертикального и горизонтального ядер
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    # Ядро размером 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Обнаружение вертикальных и горизонтальных линий
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    # Объединение горизонтальных и вертикальных линий в новом изображении
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Размывание и установление порогового значения
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Обнаружение контуров
    contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Преобразование tuple в list
    contours = list(contours)
    contours.reverse()

    # Подсчёт вертикальных линий (столбцов)
    num_vertical_lines = 1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if 20 < w < 500 and 20 < h < 500:
            num_vertical_lines = int(img.shape[1] / w)
            break

    texts = []
    # Итерация по каждому контуру (ячейке)
    for contour in contours:

        # Получение координат ограничивающего прямоугольника вокруг контура
        x, y, w, h = cv2.boundingRect(contour)
        if 20 < w < 500 and 20 < h < 500:
            cell_image = img[y:y + h, x:x + w]

            # Получение текста с ячейки
            text = pytesseract.image_to_string(cell_image, lang='rus')

            if text == '' or re.search(r'\d', text):
                results = reader.readtext(cell_image)
                if len(results) > 0:
                    text = results[0][1]

            # Добавление текста в список
            texts.append(text.replace('|', '').replace('_', '').replace(',', '.'))

    # Исключение пустых строк вначале списка
    for text in texts[num_vertical_lines:]:
        if text == '':
            texts.remove(text)
        else:
            break

    # Разделение списка по колучиству столбцов
    texts = [texts[i:i + num_vertical_lines] for i in range(0, len(texts), num_vertical_lines)]

    # Создание датафрейма
    try:
        df = pd.DataFrame(data=texts[1:], columns=texts[0])
        df.dropna(inplace=True)
        df = df.applymap(lambda x: x.replace('\n', ''))
        df = df.rename(columns=lambda x: re.sub(r'\s+', ' ', x.replace('\n', ' ')))
    except:
        df = pd.DataFrame()
    return df


def df2html_editable(df: pd.DataFrame, table_name: str):
    """
    Преобразование фрейма данных pandas в HTML-таблицу с редактируемыми ячейками
    """
    html = df.to_html(escape=False)

    js_code = f"""
    <script>
    function updateCellValue(row, col, value) {{
        // Send an API request to update the dataframe value
        // You can implement the API endpoint in your FastAPI application
        // Example: fetch('/update', {{ method: 'POST', body: JSON.stringify({{ row, col, value }}) }});
        console.log('Updating value:', row, col, value);
    }}
    const table = document.getElementById('table_{table_name}');
    for (let row of table.rows) {{
        for (let cell of row.cells) {{
            cell.contentEditable = true;
            cell.addEventListener('input', () => {{
                const rowIdx = cell.parentNode.rowIndex;
                const colIdx = cell.cellIndex;
                const value = cell.innerText;
                updateCellValue(rowIdx, colIdx, value);
            }});
        }}
    }}
    </script>
    """

    html = html.replace("<table", f"<table id='table_{table_name}'")
    return html + js_code


def create_html_content(img_path: str, filename: str):
    """
    Создание HTML
    :param img_path: путь до изобажения
    :param filename: имя файла
    :return: HTML-содержимое
    """
    df = img2table(img_path)
    if df.empty:
        html_content = f"<h2>Не удалось распознать таблицу с изображения {filename}</h2>"
    else:
        html_content = f"<h2>Таблица с изображения {filename}</h2>"
        html_content += df2html_editable(df, filename)
    return html_content
