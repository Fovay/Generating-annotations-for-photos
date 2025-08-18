from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import uvicorn
import nest_asyncio
from pyngrok import ngrok
import os
from fastapi.staticfiles import StaticFiles
from shutil import copyfile
from torchvision import transforms
from PIL import ImageEnhance, ImageFilter, ImageOps
import logging
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR
import easyocr
from googletrans import Translator
import asyncio
import random
from io import BytesIO
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from tqdm import tqdm
import evaluate
import nltk
import csv
from datetime import datetime
import pandas as pd


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# настройка статических файлов
app = FastAPI()
translator = Translator()
# папка для сохранения изображений и логов
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "generated")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "metrics_log.csv")
logo_path = "C:/Img/static/logo1.png"
if not os.path.exists(logo_path):
    raise FileNotFoundError(f"The logo image '{logo_path}' was not found.")
else:
    static_logo_path = "static/logo1.png"
    if not os.path.exists(static_logo_path):
        copyfile(logo_path, static_logo_path)

# нНастройка обслуживания статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename", "caption", "BLEU", "ROUGE-L", "METEOR"])

# загрузка предобученной модели BLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
model.eval()
model.to(device)
# processor = BlipProcessor.from_pretrained("./blip-finetuned")
# model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned").to(device)
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = AutoModelForCausalLM.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

# # функция генерации длинных аннотаций
# def generate_caption(image: Image.Image, word_count: int) -> str:
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=word_count*2,  # ограничение длины
#             num_beams=8,            # улучшает связность
#             temperature=random.uniform(0.7, 1.2),  # случайная температура для разнообразия
#             top_k=50,  # ограничиваем выбор топ-50 слов для разнообразия
#             repetition_penalty=1.2, # иИзбегает повторений
#             length_penalty=1.5,     # пПоощряет длинные описания
#             no_repeat_ngram_size=1   # блокирует повторяющиеся фразы
#         )
#     return processor.decode(generated_ids[0], skip_special_tokens=True)


async def generate_caption(image: Image.Image, word_count: int) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)

    # увеличиваем для более длинных описаний
    max_tokens = max(100, word_count * 3)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,  # мак число токенов
            min_new_tokens=word_count // 2,  # мин длина для стабильности
            num_beams=10,  # для большей связности
            temperature=0.9,  # температура для баланса
            top_k=100,  # для большего разнообразия
            top_p=0.95,  # для естественности
            repetition_penalty=1.3,  # для избегания повторов
            length_penalty=2.0,  # поощряем более длинные описания
            no_repeat_ngram_size=2,  # предотвращаем повторение биграмм
            do_sample=True,  # семплирование для разнообразия
            early_stopping=False,  # отключаем раннюю остановку
        )

    # декодируем результат
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # удаляем лишние пробелы и очищаем текст
    caption = " ".join(caption.split()).strip()

    return caption


def evaluate_caption(generated_caption: str, references: list[str]) -> dict:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    try:
        bleu_score = bleu.compute(predictions=[generated_caption], references=[references])
        rouge_score = rouge.compute(predictions=[generated_caption], references=[references])
        meteor_score = meteor.compute(predictions=[generated_caption], references=[references])

        bleu_val = round(bleu_score["bleu"] * 100, 2)
        rouge_val = round(rouge_score["rougeL"] * 100, 2)
        meteor_val = round(meteor_score["meteor"] * 100, 2)

        print(f"Оценка сгенерированного описания:")
        print(f"BLEU: {bleu_val}%")
        print(f"ROUGE-L: {rouge_val}%")
        print(f"METEOR: {meteor_val}%\n")

        return {"BLEU": bleu_val, "ROUGE-L": rouge_val, "METEOR": meteor_val}

    except Exception as e:
        print("Ошибка при вычислении метрик:", e)
        return {"BLEU": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}


# OCR
ocr_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-large-printed", do_rescale=False)
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(device)

# EasyOCR
reader = easyocr.Reader(["ru", "en"], gpu=torch.cuda.is_available())


# предобработки изображения
def preprocess_image(image: Image.Image) -> np.ndarray:
    # конвертация в RGB
    image = image.convert("RGB")
    np_image = np.array(image)

    # адаптивное масштабирование
    height, width = np_image.shape[:2]
    scale = max(2, 1200 / min(height, width))
    np_image = cv2.resize(np_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # улучшение резкости и контраста
    image_pil = Image.fromarray(np_image)
    image_pil = ImageEnhance.Sharpness(image_pil).enhance(3.0)
    image_pil = ImageEnhance.Contrast(image_pil).enhance(2.0)
    np_image = np.array(image_pil)

    # преобразование в градации серого
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    # для точного выделения текста
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # для улучшения мелких деталей
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # удаление шума
    thresh = cv2.fastNlMeansDenoising(thresh, h=10)

    return thresh


# мин постобработки (только очистка)
def postprocess_text(text: str) -> str:
    # убираем только лишние пробелы
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines) if len(lines) > 1 else " ".join(lines)


# распознавания текста с EasyOCR
def ocr_text(image: Image.Image) -> str:
    try:
        # предобработка изображения
        processed_image = preprocess_image(image)
        result = reader.readtext(processed_image, detail=1,
            width_ths=0.8,  # разделения слов
            mag_ratio=2.0,  # внутреннее масштабирование
            text_threshold=0.7,  # распознавания
        )

        # сортировка по y-координате для сохранения порядка строк
        result = sorted(result, key=lambda x: x[0][0][1])

        # сборка текста с учетом координат
        lines = []
        prev_y = None
        current_line = []

        for bbox, text, prob in result:
            y = bbox[0][1]  # y-координата верхней точки текста
            x = bbox[0][0]  # x-координата для горизонтального порядка

            # новая строка, если разница по y больше порога
            if prev_y is None or abs(y - prev_y) > 25:  # порог для разделения строк
                if current_line:
                    current_line.sort(key=lambda t: t[1])
                    lines.append(" ".join(t[0] for t in current_line))
                current_line = [(text, x)]
            else:
                current_line.append((text, x))

            prev_y = y

        if current_line:
            current_line.sort(key=lambda t: t[1])
            lines.append(" ".join(t[0] for t in current_line))

        # объединение строк
        raw_text = "\n".join(lines)

        # постобработка
        final_text = postprocess_text(raw_text)

        return final_text if final_text else "Текст не обнаружен"
    except Exception as e:
        logging.error(f"OCR error: {str(e)}")
        return f"Ошибка распознавания: {str(e)}"


@app.get("/metrics_data")
def get_metrics():
    df = pd.read_csv(log_path).dropna()
    return df.to_dict(orient="records")


# Главная страница
@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Генерация аннотаций</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        <style>
            body {
                font-family: 'Orbitron', sans-serif;
                text-align: center;
                padding: 20px;
                background-color: #f7f7f7;
                margin: 0;
            }
            .navbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                background-color: #fff;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                position: fixed;
                width: 100%;
                top: 0;
                z-index: 1000;
                left: 0;
            }
            .navbar .logo {
                display: flex;
                align-items: center;
            }
            .navbar .logo img {
                width: 15%;
                margin-right: 10px;
            }
            .navbar .nav-links {
                display: flex;
                gap: 20px;
                margin-left: 20px;
            }
            .navbar .nav-links a {
                color: #333;
                text-decoration: none;
                font-weight: bold;
                font-size: 18px;
                transition: color 0.3s ease, transform 0.3s ease;
            }
            .navbar .nav-links a:hover {
                color: orange;
                transform: scale(1.1);
            }
            .navbar .nav-links a.annotation {
                color: black;
                text-decoration: none;
            }
            .navbar .nav-links a.image-text {
                color: orange;
            }
            .content {
                max-width: 1200px;
                margin: 80px auto 20px;
                padding: 20px;
                background-color: #f7f7f7;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
            .control-button {
                display: inline-block;
                padding: 10px 20px;
                margin: 5px;
                background-color: #fff;
                border: 2px solid #ccc;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            .control-button:hover {
                background-color: #f0f0f0;
                transform: scale(1.05);
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            }
            .control-button select, .control-button input {
                border: none;
                background: none;
                font-family: 'Orbitron', sans-serif;
                font-size: 16px;
                width: 100px;
                padding: 5px;
                outline: none;
            }
            #language {
                width: 150px;
            }
            #dropzone {
                border: 2px dashed #ccc;
                padding: 40px;
                width: 80%;
                margin: 20px auto;
                border-radius: 15px;
                font-size: 18px;
                cursor: pointer;
                color: #666;
                background-color: #fdfdfd;
                transition: background-color 0.3s ease, transform 0.3s ease;
            }
            #dropzone:hover {
                background-color: #f0f0f0;
                transform: scale(1.05);
            }
            #dropzone p {
                margin: 0;
                font-weight: bold;
            }
            #uploadedImage {
                display: none;
                margin: 20px auto;
                max-width: 100%;
                max-height: 600px;
                object-fit: contain;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
                animation: fadeIn 1s ease-in-out;
            }
            #progressContainer {
                margin: 20px auto;
                width: 80%;
                position: relative;
                display: none;
            }
            #progressBar {
                width: 0%;
                height: 20px;
                background: linear-gradient(to right, orange, #ff7f00);
                border-radius: 10px;
                transition: width 0.3s ease;
            }
            #progressText {
                position: absolute;
                top: 0;
                left: 50%;
                transform: translateX(-50%);
                font-weight: bold;
                color: #333;
            }
            #result {
                margin: 20px auto;
                width: 80%;
                padding: 15px;
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                display: none;
                animation: fadeIn 1s ease-in-out;
                position: relative;
                text-align: center;
            }
            #annotationText {
                width: 100%;
                height: 150px;
                resize: none;
                border: 1px solid #ccc;
                padding: 10px;
                font-size: 16px;
                box-sizing: border-box;
                font-family: 'Orbitron', sans-serif;
            }
            #editButton, #saveButton {
                padding: 5px 10px;
                margin: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
                cursor: pointer;
                background-color: #fff;
                transition: background-color 0.3s ease;
            }
            #editButton:hover, #saveButton:hover {
                background-color: #f0f0f0;
            }
            #copyButton {
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px 10px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.3s ease;
            }
            #copyButton:hover {
                background-color: #f0f0f0;
            }
            #exportButton, #reloadButton {
                display: inline-block;
                padding: 10px 20px;
                margin: 10px;
                background: linear-gradient(to right, #ff7f00, #ff4500);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            }
            #exportButton:hover, #reloadButton:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
            }
            #feedback {
                margin: 20px auto;
                display: none;
            }
            #feedback button {
                font-size: 30px;
                padding: 10px 20px;
                margin: 10px 5px;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease, transform 0.3s ease;
            }
            #feedback button:hover {
                background-color: #f0f0f0;
                transform: scale(1.1);
            }
            #featuresBlock {
                max-width: 1200px;
                margin: 20px auto 80px auto;
                padding: 20px;
                background-color: #f7f7f7;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .features-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                max-width: 900px;
                width: 100%;
            }
            .feature {
                background-color: #fdfdfd;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }
            .feature:hover {
                transform: scale(1.05);
            }
            .feature i {
                font-size: 30px;
                color: orange;
                margin-bottom: 10px;
            }
            .feature h3 {
                font-size: 18px;
                margin-bottom: 10px;
                color: #333;
            }
            .feature p {
                font-size: 14px;
                color: #666;
            }
            .notice {
                font-size: 16px;
                color: #888;
                margin-top: 15px;
            }
            .footer {
                text-align: center;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0px -4px 10px rgba(0, 0, 0, 0.1);
                width: 100%;
                position: fixed;
                bottom: 0;
                left: 0;
            }
            .menu-icon {
                display: none;
                cursor: pointer;
            }
            @media (max-width: 768px) {
                .menu-icon {
                    display: block;
                }
                .navbar .nav-links {
                    display: none;
                    flex-direction: column;
                    position: absolute;
                    top: 60px;
                    left: 0;
                    width: 100%;
                    background-color: #fff;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                }
                .navbar .nav-links.open {
                    display: flex;
                }
                .navbar .nav-links a {
                    margin: 10px 0;
                    padding: 10px;
                }
                .features-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            @media (max-width: 480px) {
                .features-grid {
                    grid-template-columns: 1fr;
                }
            }
            h1 {
                font-size: 40px;
                font-weight: bold;
                color: #333;
                animation: fadeIn 1s ease-in-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            #exportModal div {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                text-align: center;
            }
            #exportModal button {
                margin: 10px;
                padding: 10px 20px;
                background: #ff7f00;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.3s ease;
            }
            #exportModal button:hover {
                background: #ff4500;
            }
        </style>
    </head>
    <body>
        <div class="navbar">
            <div class="logo">
                <img src="/static/logo1.png" alt="Logo">
                <div class="nav-links">
                    <a href="/" class="nav-link annotation" onclick="navigateTo(event, '/')">Генерация аннотаций</a>
                    <a href="/ocr" class="nav-link image-text" onclick="navigateTo(event, '/ocr')">Считать текст с изображения</a>
                </div>
            </div>
            <div class="menu-icon" onclick="toggleMenu()">☰</div>
        </div>
        <div class="content">
            <h1>Генерация аннотаций</h1>
            <div class="control-button" id="languageSelect">
                <label for="language">Выберите язык аннотации:</label>
                <select id="language">
                    <option value="en">Английский</option>
                    <option value="ru">Русский</option>
                    <option value="de">Немецкий</option>
                </select>
            </div>
            <div class="control-button" id="wordCountContainer">
                <label for="wordCount">Количество слов в аннотации:</label>
                <input type="number" id="wordCount" value="50" min="10" max="200">
            </div>
            <div id="dropzone">
                <p>Перетащите изображение сюда или кликните, чтобы выбрать</p>
                <p class="notice">Разрешенные форматы: JPEG, PNG</p>
            </div>
            <input type="file" id="fileInput" accept="image/jpeg, image/png" style="display: none;">
            <img id="uploadedImage" class="fade-in">
            <div id="progressContainer">
                <div id="progressBar"></div>
                <span id="progressText">0%</span>
            </div>
            <div id="result" class="fade-in">
                <textarea id="annotationText" readonly></textarea>
                <button id="editButton">Редактировать</button>
                <button id="saveButton" style="display: none;">Сохранить</button>
                <button id="copyButton" onclick="copyToClipboard()"><i class="fas fa-copy"></i></button>
            </div>
            <div id="feedback" style="display: none;">
                <button onclick="like()">👍</button>
                <button onclick="dislike()">👎</button>
            </div>
            <div id="actionButtons" style="display: none;">
                <button id="exportButton">Экспорт</button>
                <button id="reloadButton">Загрузить новое изображение</button>
            </div>
        </div>
        <div id="featuresBlock">
            <h2>Функционал веб-сервиса</h2>
            <div class="features-grid">
                <div class="feature">
                    <i class="fas fa-upload"></i>
                    <h3>Простая загрузка</h3>
                    <p>Легко загружайте изображения в поддерживаемых форматах: JPEG, PNG.</p>
                </div>
                <div class="feature">
                    <i class="fas fa-brain"></i>
                    <h3>Мощный ИИ</h3>
                    <p>ИИ анализирует изображения для точных описаний.</p>
                </div>
                <div class="feature">
                    <i class="fas fa-edit"></i>
                    <h3>Редактирование</h3>
                    <p>Изменяйте аннотации по вашим нуждам.</p>
                </div>
                <div class="feature">
                    <i class="fas fa-file-export"></i>
                    <h3>Экспорт</h3>
                    <p>Сохраняйте аннотации в разных форматах.</p>
                </div>
                <div class="feature">
                    <i class="fas fa-mobile-alt"></i>
                    <h3>Мобильность</h3>
                    <p>Работайте с любого устройства.</p>
                </div>
                <div class="feature">
                    <i class="fas fa-lock"></i>
                    <h3>Безопасность</h3>
                    <p>Ваши данные защищены.</p>
                </div>
            </div>
        </div>
        <div class="footer">
            © IMAGEN 2025<sup>®</sup>
        </div>
        <div id="exportModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); justify-content: center; align-items: center;">
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                <h2>Выберите формат экспорта</h2>
                <button onclick="exportTo('txt')">TXT</button>
                <button onclick="exportTo('docx')">DOCX</button>
                <button onclick="exportTo('pdf')">PDF</button>
                <button onclick="closeModal()">Отмена</button>
            </div>
        </div>
        <script>
            const editButton = document.getElementById("editButton");
            const saveButton = document.getElementById("saveButton");
            const annotationTextArea = document.getElementById("annotationText");

            editButton.addEventListener("click", () => {
                annotationTextArea.removeAttribute("readonly");
                editButton.style.display = "none";
                saveButton.style.display = "inline-block";
            });

            saveButton.addEventListener("click", () => {
                annotationTextArea.setAttribute("readonly", true);
                saveButton.style.display = "none";
                editButton.style.display = "inline-block";
            });

            function navigateTo(event, path) {
                event.preventDefault();
                window.history.pushState({}, '', path);
                fetch(path)
                    .then(response => response.text())
                    .then(html => {
                        document.open();
                        document.write(html);
                        document.close();
                    })
                    .catch(error => console.error('Error:', error));
            }
            const dropzone = document.getElementById("dropzone");
            const fileInput = document.getElementById("fileInput");
            const uploadedImage = document.getElementById("uploadedImage");
            const progressContainer = document.getElementById("progressContainer");
            const progressBar = document.getElementById("progressBar");
            const progressText = document.getElementById("progressText");
            const result = document.getElementById("result");
            const annotationText = document.getElementById("annotationText");
            const feedbackSection = document.getElementById("feedback");
            const copyButton = document.getElementById("copyButton");
            const exportButton = document.getElementById("exportButton");
            const reloadButton = document.getElementById("reloadButton");
            const languageSelect = document.getElementById("language");
            const wordCountInput = document.getElementById("wordCount");
            const actionButtons = document.getElementById("actionButtons");

            dropzone.addEventListener("click", () => fileInput.click());
            dropzone.addEventListener("dragover", (e) => {
                e.preventDefault();
                dropzone.style.backgroundColor = "#f0f0f0";
            });
            dropzone.addEventListener("dragleave", () => {
                dropzone.style.backgroundColor = "#fdfdfd";
            });
            dropzone.addEventListener("drop", (e) => {
                e.preventDefault();
                dropzone.style.backgroundColor = "#fdfdfd";
                const file = e.dataTransfer.files[0];
                handleFile(file);
            });

            fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

            function handleFile(file) {
                if (!file.type.match("image/jpeg") && !file.type.match("image/png")) {
                    alert("Пожалуйста, загрузите изображение в формате JPEG или PNG!");
                    return;
                }

                const reader = new FileReader();
                reader.onload = () => {
                    uploadedImage.src = reader.result;
                    uploadedImage.style.display = "block";
                    dropzone.style.display = "none";
                };
                reader.readAsDataURL(file);

                result.style.display = "none";
                feedbackSection.style.display = "none";
                actionButtons.style.display = "none";
                progressContainer.style.display = "block";
                progressBar.style.width = "0%";
                progressText.innerText = "0%";
                let progress = 0;
                const interval = setInterval(() => {
                    progress += 1;
                    progressBar.style.width = progress + "%";
                    progressText.innerText = progress + "%";
                    if (progress >= 100) clearInterval(interval);
                }, 300);

                const formData = new FormData();
                formData.append("file", file);
                formData.append("language", languageSelect.value);
                formData.append("word_count", wordCountInput.value);

                fetch("/predict_caption", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    clearInterval(interval);
                    progressBar.style.width = "100%";
                    progressText.innerText = "100%";
                    setTimeout(() => {
                        progressContainer.style.display = "none";
                        annotationText.value = data.image_caption;
                        result.style.display = "block";
                        feedbackSection.style.display = "block";
                        copyButton.style.display = "block";
                        actionButtons.style.display = "flex";
                        actionButtons.style.justifyContent = "center";
                    }, 500);
                })
                .catch(error => {
                    console.error("Ошибка:", error);
                    progressContainer.style.display = "none";
                    annotationText.value = "Ошибка при генерации аннотации.";
                    result.style.display = "block";
                });
            }

            function copyToClipboard() {
                navigator.clipboard.writeText(annotationText.value)
                    .then(() => {})
                    .catch(err => console.error("Ошибка копирования:", err));
            }

            function showExportModal() {
                document.getElementById('exportModal').style.display = 'flex';
            }

            function closeModal() {
                document.getElementById('exportModal').style.display = 'none';
            }

            function exportTo(format) {
                fetch(`/export_annotation?format=${format}&text=${encodeURIComponent(annotationText.value)}`)
                    .then(response => response.blob())
                    .then(blob => {
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `annotation.${format}`;
                        a.click();
                        window.URL.revokeObjectURL(url);
                        closeModal();
                    })
                    .catch(error => console.error("Ошибка экспорта:", error));
            }

            exportButton.addEventListener("click", showExportModal);

            reloadButton.addEventListener("click", () => {
                uploadedImage.style.display = "none";
                result.style.display = "none";
                feedbackSection.style.display = "none";
                actionButtons.style.display = "none";
                dropzone.style.display = "block";
                fileInput.value = "";
            });

            function toggleMenu() {
                const navLinks = document.querySelector('.navbar .nav-links');
                navLinks.classList.toggle('open');
            }

            function like() {
                sendFeedback('like');
            }

            function dislike() {
                sendFeedback('dislike');
            }

            function sendFeedback(feedback) {
                fetch("/feedback", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ feedback: feedback })
                })
                .then(response => response.json())
                .then(data => console.log("Feedback received:", data))
                .catch(error => console.error("Error sending feedback:", error));
            }
        </script>
        # <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        # <script>
        # async function fetchMetrics() {
        #     const res = await fetch("/metrics_data");
        #     const data = await res.json();
        #     const labels = data.map(x => x.filename);
        #     const bleu = data.map(x => x.BLEU);
        #     const rouge = data.map(x => x["ROUGE-L"]);
        #     const meteor = data.map(x => x.METEOR);
            
        #     metricsChart.data.labels = labels;
        #     metricsChart.data.datasets[0].data = bleu;
        #     metricsChart.data.datasets[1].data = rouge;
        #     metricsChart.data.datasets[2].data = meteor;
        #     metricsChart.update();
        # }
        # const canvas = document.getElementById("metricsChart");
        # const ratio = window.devicePixelRatio || 1;
        # canvas.width = canvas.clientWidth * ratio;
        # canvas.height = canvas.clientHeight * ratio;
        # canvas.getContext("2d").scale(ratio, ratio);
        # const ctx = canvas.getContext("2d");
        # const metricsChart = new Chart(ctx, {  
        #             type: "bar",
        #     data: {
        #         labels: [],
        #         datasets: [
        #             {
        #                 label: "BLEU",
        #                 data: [],
        #                 backgroundColor: "rgba(255, 127, 0, 0.5)",
        #                 borderColor: "rgba(255, 127, 0, 1)",
        #                 borderWidth: 1
        #             },
        #             {
        #                 label: "ROUGE-L",
        #                 data: [],
        #                 backgroundColor: "rgba(0, 150, 136, 0.5)",
        #                 borderColor: "rgba(0, 150, 136, 1)",
        #                 borderWidth: 1
        #             },
        #             {
        #                 label: "METEOR",
        #                 data: [],
        #                 backgroundColor: "rgba(33, 150, 243, 0.5)",
        #                 borderColor: "rgba(33, 150, 243, 1)",
        #                 borderWidth: 1
        #             }
        #         ]
        #     },
        #     options: {
        #         responsive: true,
        #         maintainAspectRatio: false,
        #         scales: {
        #             y: {
        #                 beginAtZero: true, 
        #                 max: 100
                        
        #             }
        #         },
        #         plugins: {
        #             legend: {
        #                 position: 'top',
        #             },
        #             title: {
        #                 display: true,
        #                 text: 'Метрики генерации аннотаций'
        #             }
        #         }
        #     }
        # });
        # setInterval(fetchMetrics, 5000);
        # </script>
    </body>
    </html>
    """


# FastAPI endpoint /ocr
@app.get("/ocr", response_class=HTMLResponse)
async def ocr_page():
    return """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Считать текст с изображения</title>
            <style>
                body {
                    font-family: 'Orbitron', sans-serif;
                    text-align: center;
                    padding: 20px;
                    background-color: #f7f7f7;
                    margin: 0;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 1rem;
                }
                @media (max-width: 600px) {
                    h2, h3 {
                        font-size: 1.2rem;
                    }
                    input, button, label {
                        font-size: 1rem;
                        width: 100%;
                        box-sizing: border-box;
                    }
                    form {
                        display: flex;
                        flex-direction: column;
                        gap: 0.5rem;
                    }
                }
                .navbar {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 20px;
                    background-color: #fff;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    position: fixed;
                    width: 100%;
                    top: 0;
                    z-index: 1000;
                    left: 0;
                }
                .navbar .logo {
                    display: flex;
                    align-items: center;
                }
                .navbar .logo img {
                    width: 15%;
                    margin-right: 10px;
                }
                .navbar .nav-links {
                    display: flex;
                    gap: 20px;
                    margin-left: 20px;
                }
                .navbar .nav-links a {
                    color: #333;
                    text-decoration: none;
                    font-weight: bold;
                    font-size: 18px;
                    transition: color 0.3s ease;
                }
                .navbar .nav-links a.active {
                    color: orange;
                    border-bottom: 2px solid orange;
                }
                .content {
                    max-width: 1200px;
                    margin: 0px auto;
                    padding: 20px;
                    background-color: #f7f7f7;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    margin-top: 80px;
                    margin-bottom: 80px;
                }
                #ocrDropzone {
                    border: 2px dashed #ccc;
                    padding: 40px;
                    width: 80%;
                    margin: 20px auto;
                    border-radius: 15px;
                    font-size: 18px;
                    cursor: pointer;
                    color: #666;
                    background-color: #fdfdfd;
                    transition: background-color 0.3s ease, transform 0.3s ease;
                }
                #ocrDropzone:hover {
                    background-color: #f0f0f0;
                    transform: scale(1.05);
                }
                #ocrDropzone p {
                    margin: 0;
                    font-weight: bold;
                }
                #ocrImage {
                    display: block;
                    margin: 20px auto;
                    max-width: 100%;
                    max-height: 600px;
                    object-fit: contain;
                    border-radius: 10px;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
                    animation: fadeIn 1s ease-in-out;
                }
                #ocrResult {
                    margin-top: 20px;
                    font-size: 18px;
                    font-weight: bold;
                    text-align: left;
                    padding-left: 20px;
                    text-align: center;
                    animation: fadeIn 1s ease-in-out;
                }
                .footer {
                    text-align: center;
                    padding: 20px;
                    background-color: #fff;
                    box-shadow: 0px -4px 10px rgba(0, 0, 0, 0.1);
                    width: 100%;
                    left: 0;
                    position: fixed;
                    bottom: 0;
                }
                .menu-icon {
                    display: none;
                    cursor: pointer;
                }
                @media (max-width: 768px) {
                    .menu-icon {
                        display: block;
                    }
                    .navbar .nav-links {
                        display: none;
                        flex-direction: column;
                        position: absolute;
                        top: 60px;
                        left: 0;
                        width: 100%;
                        background-color: #fff;
                        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    }
                    .navbar .nav-links.open {
                        display: flex;
                    }
                    .navbar .nav-links a {
                        margin: 10px 0;
                        padding: 10px;
                    }
                }
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
            </style>
        </head>
        <body>
            <div class="navbar">
                <div class="logo">
                    <img src="/static/logo1.png" alt="Logo">
                    <div class="nav-links">
                        <a href="/" class="nav-link">Генерация аннотаций</a>
                        <a href="/ocr" class="nav-link">Считать текст с изображения</a>
                    </div>
                </div>
                <div class="menu-icon" onclick="toggleMenu()">
                    ☰
                </div>
            </div>
            <div class="content">
                <h1>Считать текст с изображения</h1>
                <div id="ocrDropzone">
                    <p>Перетащите изображение сюда или кликните, чтобы выбрать</p>
                    <p class="notice">Разрешенные форматы: JPEG, PNG</p>
                </div>
                <input type="file" id="ocrFileInput" accept="image/jpeg, image/png" style="display: none;">
                <img id="ocrImage" style="display: none;">
                <p id="ocrResult"></p>
            </div>
            <div class="footer">
                &copy; IMAGEN 2025<sup>®</sup>
            </div>
            <script>
                const ocrDropzone = document.getElementById("ocrDropzone");
                const ocrFileInput = document.getElementById("ocrFileInput");
                const ocrResult = document.getElementById("ocrResult");
                const ocrImage = document.getElementById("ocrImage");

                ocrDropzone.addEventListener("click", () => ocrFileInput.click());

                ocrDropzone.addEventListener("dragover", (e) => {
                    e.preventDefault();
                    ocrDropzone.style.backgroundColor = "#f0f0f0";
                });

                ocrDropzone.addEventListener("dragleave", () => {
                    ocrDropzone.style.backgroundColor = "#fdfdfd";
                });

                ocrDropzone.addEventListener("drop", (e) => {
                    e.preventDefault();
                    ocrDropzone.style.backgroundColor = "#fdfdfd";
                    const file = e.dataTransfer.files[0];
                    handleFile(file);
                });

                ocrFileInput.addEventListener("change", (e) => {
                    const file = e.target.files[0];
                    handleFile(file);
                });

                function handleFile(file) {
                    const fileType = file.type;
                    if (!fileType.includes("image/jpeg") && !fileType.includes("image/png")) {
                        alert("Пожалуйста, загрузите изображение в формате JPEG или PNG!");
                        return;
                    }

                    const reader = new FileReader();
                    reader.onloadend = function () {
                        ocrImage.src = reader.result;
                        ocrImage.style.display = "block";
                    };
                    reader.readAsDataURL(file);

                    const formData = new FormData();
                    formData.append("file", file);

                    fetch("/extract_text", {
                        method: "POST",
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        ocrResult.innerText = data.text || "Текст не обнаружен";
                    })
                    .catch(error => {
                        console.error("Ошибка:", error);
                        ocrResult.innerText = "Ошибка при распознавании текста.";
                    });
                }

                function toggleMenu() {
                    const navLinks = document.querySelector('.navbar .nav-links');
                    navLinks.classList.toggle('open');
                }
            </script>
        </body>
        </html>
        """


# API для обработки изображения и генерации аннотации
@app.post("/predict_caption")
async def predict_caption(
    file: UploadFile = File(...), language: str = Form(...), word_count: int = Form(50)
):
    try:
        # считываем байты и открываем изображение
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # проверка размера изображения
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(
                status_code=400,
                detail="Изображение слишком маленькое (минимальный размер 50x50 пикселей)",
            )

        # проверка word_count
        if word_count < 10 or word_count > 200:
            raise HTTPException(
                status_code=400, detail="Количество слов должно быть от 10 до 200"
            )

        caption = await generate_caption(image, word_count)

        # эталоны для метрик
        references = [
            "A singer in a black suit stands near a microphone as an orchestra plays in the background",
            "A man in a tuxedo sings into a microphone on stage with a band playing behind him.",
            "A musical performance is happening with orchestra and solo singer in formal clothes.",
            "A performer in a black suit holds a mic while the orchestra plays on stage.",
        ]

        # оценка метрик
        scores = evaluate_caption(caption, references)
        logger.info(
            f"Метрики оценки: BLEU = {scores['BLEU']} | ROUGE-L = {scores['ROUGE-L']}"
        )

        # сохраняем изображение и лог
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        img_path = f"generated/{timestamp}_{file.filename}"
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        with open(log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    file.filename,
                    caption,
                    scores["BLEU"],
                    scores["ROUGE-L"],
                    scores["METEOR"],
                ]
            )

        # перевод
        if language in ["ru", "de"]:
            translated = translator.translate(caption, src="en", dest=language)
            translated_caption = translated.text
        else:
            translated_caption = caption

        # ответ
        return JSONResponse(content={"image_caption": translated_caption})

    except Exception as e:
        logger.error(f" Ошибка при генерации аннотации: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/export_annotation")
async def export_annotation(format: str = Query(...), text: str = Query(...)):
    try:
        if format == "txt":
            # экспорт в текстовый файл
            return StreamingResponse(
                iter([text]),
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=annotation.txt"},
            )
        elif format == "docx":
            # экспорт в .docx
            doc = Document()
            doc.add_paragraph(text)
            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return StreamingResponse(
                buffer,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": "attachment; filename=annotation.docx"},
            )
        elif format == "pdf":
            # экспорт в .pdf
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.drawString(100, 750, text)
            p.showPage()
            p.save()
            buffer.seek(0)
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=annotation.pdf"},
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Неподдерживаемый формат. Доступны: txt, docx, pdf",
            )
    except Exception as e:
        logger.error(f"Ошибка при экспорте аннотации: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# API для извлечения текста
@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        text = ocr_text(image)
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": f"Ошибка обработки: {str(e)}"}, status_code=500)


# запуск сервера ngrok
if __name__ == "__main__":
    try:
        NGROK_AUTH_TOKEN = ""
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(8000).public_url
        print(f"Сервер доступен по адресу: {public_url}")

    except Exception as e:
        print(f"Ошибка при запуске ngrok: {e}")
        exit(1)
        
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
