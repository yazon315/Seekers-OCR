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
import dateparser
import datetime
from ultralyticsplus import YOLO
import imutils

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
reader = easyocr.Reader(['ru'])

model = YOLO('keremberke/yolov8m-table-extraction')

model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image


def remove_images():
    """
        Убирание прошлых изображений из папки
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
    array = np.array(img)

    try:
        results = model.predict(img)

        for result in results:
            bbox = result.boxes.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = bbox
            reserve = int(array.shape[0] / array.shape[1] * (x_max / x_min))
            reserve2 = int(array.shape[0] / array.shape[1] * (y_max / x_min))
            cropped_image = img.crop((int(x_min) - reserve,
                                      int(y_min) - reserve,
                                      int(x_max) + reserve + reserve2,
                                      int(y_max) + reserve + reserve2))
            cropped_image = np.array(cropped_image)
    except:
        if array.ndim == 3 and array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        # Присвоение изображению порогового значения в виде двоичного изображения
        img_bin = cv2.adaptiveThreshold(array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        # Обнаружение контуров
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            new_contours = []
            for contour in contours:
                _, _, w, h = cv2.boundingRect(contour)
                if 0.6 < w / array.shape[1] < 1 and 0 < h / array.shape[0] < 0.4:
                    new_contours.append(contour)

            # Определение самого большого контура
            largest_contour = max(new_contours, key=cv2.contourArea)
        except:
            largest_contour = max(contours, key=cv2.contourArea)

        # Обрезание по этому контуру
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = array[y:y + h, x:x + w]

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

    def sort_contours_yx(cnts, y_thresh=5):
        # Создание списка ограничивающих прямоугольников для каждого контура
        boxes = [cv2.boundingRect(c) for c in cnts]

        # Сортирука контуров по оси y
        sorted_boxes = sorted(boxes, key=lambda b: b[1])

        sorted_cnts = [x for _, x in sorted(zip(boxes, cnts), key=lambda pair: pair[0][1])]

        # Инициализация первого значения y и списка для сортировки по x
        last_y = sorted_boxes[0][1]
        current_group = []
        final_cnts = []

        for (x, y, w, h), c in zip(sorted_boxes, sorted_cnts):
            if abs(y - last_y) > y_thresh:  # Проверка разницы с предыдущим y
                # Сортировка текущей группы по x и сохранение результатов
                current_group.sort(key=lambda b: b[0])
                # Добавление отсортированной группы в финальный список
                final_cnts += current_group
                current_group = []
            current_group.append(((x, y, w, h), c))
            last_y = y

        # Не забываем про последнюю группу
        if current_group:
            current_group.sort(key=lambda b: b[0][0])
            final_cnts += current_group

        # Разделение контуров и боксов обратно в два списка
        final_boxes, final_cnts = zip(*final_cnts)

        return final_cnts, final_boxes

    # Считывание изображения
    img = Image.open(img_path)
    img = np.array(img)
    cv2_image = imutils.resize(img, height=500)

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск контуров
    edged = cv2.Canny(img, 75, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    grabbed = imutils.grab_contours(contours)
    sortedContours = sorted(grabbed, key=cv2.contourArea, reverse=True)[:5]

    # Выделение максимального
    screenCnt = max(sortedContours, key=cv2.contourArea)

    # Присвоение изображению порогового значения в виде двоичного изображения
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    try:
        # Исправление перспективы изображения
        ratio = cv2_image.shape[0] / 500.0
        pts = screenCnt.reshape(4, 2) * ratio

        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        raw_transformed = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        bin_transformed = cv2.warpPerspective(img_bin, M, (maxWidth, maxHeight))
    except:
        raw_transformed = img
        bin_transformed = img_bin

    # Ширина ядра как 100-я часть общей ширины
    kernel_len = np.array(img).shape[1] // 100

    # Определение вертикального и горизонтального ядер
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    # Ядро размером 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Обнаружение вертикальных и горизонтальных линий
    image_1 = cv2.erode(bin_transformed, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    image_2 = cv2.erode(bin_transformed, hor_kernel, iterations=3)
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
    contours, _ = sort_contours_yx(contours, 5)

    texts = []
    # Итерация по каждому контуру (ячейке)
    for contour in contours:

        # Получение координат ограничивающего прямоугольника вокруг контура
        x, y, w, h = cv2.boundingRect(contour)
        if 0.04 < w / raw_transformed.shape[1] < 0.35 and 0.04 < h / raw_transformed.shape[0] < 0.35:
            cell_image = raw_transformed[y:y + h, x:x + w]

            # Получение текста с ячейки
            text = pytesseract.image_to_string(cell_image, lang='rus')

            if text == '' or re.search(r'\d', text):
                results = reader.readtext(cell_image)
                if len(results) > 0:
                    text = results[0][1]

            # Добавление текста в список
            texts.append(text.replace('|', '').replace('_', '').replace(',', '.').replace('—', '').replace('\n', ''))

    # Создание итогового датафрейма
    dates = []
    date_ids = []
    for i in range(len(texts)):
        date = dateparser.parse(texts[i], languages=['ru'], date_formats=['%d/%m/%Y'],
                                settings={'REQUIRE_PARTS': ['day', 'month', 'year']})
        if type(date) == datetime.datetime and datetime.datetime(2000, 1, 1) < date <= datetime.datetime.now() \
                and i != len(texts) - 1 and i != len(texts) - 2:
            dates.append(date.strftime('%d.%m.%Y'))
            date_ids.append(i)

    kinds = []
    types = []
    quantities = []
    df = pd.DataFrame(columns=['Дата донации', 'Класс крови', 'Тип донации', 'Количество'])

    if date_ids:
        for i in date_ids:
            if 'бв' in texts[i + 1] or 'б' in texts[i + 1] or 'в' in texts[i + 1]:
                types.append('Безвозмездно')
            elif 'платно' in texts[i + 1]:
                types.append('Платно')
                texts[i + 1].replace('платно', '')
            else:
                types.append('')

            if 'кр' in texts[i + 1] or 'к' in texts[i + 1] or 'р' in texts[i + 1]:
                kinds.append('Цельная кровь')
            elif 'пл' in texts[i + 1] or 'п' in texts[i + 1] or 'л' in texts[i + 1]:
                kinds.append('Плазма')
            elif 'ц' in texts[i + 1] or 'т' in texts[i + 1]:
                kinds.append('Тромбоциты')
            else:
                kinds.append('')

            if texts[i + 2] != '':
                quantities.append(re.sub(r"\D", "", texts[i + 2]))
            else:
                quantities.append('')

        df['Дата донации'] = dates
        df['Класс крови'] = kinds
        df['Тип донации'] = types
        df['Количество'] = quantities

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
