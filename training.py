import os
import json
import random
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
JSON_PATH = "C:/Img/dataset.json"
IMG_DIR = "C:/Img/img"
IMG_DIR = os.path.normpath(IMG_DIR)
assert os.path.exists(JSON_PATH), f"JSON файл не найден: {JSON_PATH}"
assert os.path.exists(IMG_DIR), f"Папка с изображениями не найдена: {IMG_DIR}"
class ImageCaptionDataset(Dataset):
    def __init__(self, json_file, img_dir, processor, max_length=512):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
        # проверка структуры данных
        valid_count = sum(1 for item in self.data 
                         if 'image' in item and 'caption' in item)
        if valid_count != len(self.data):
            logger.warning(f"Некорректные записи: {len(self.data)-valid_count}/{len(self.data)}")
        # проверка существования файлов
        self.valid_data = []
        for item in self.data:
            if 'image' in item and 'caption' in item:
                image_path = os.path.normpath(os.path.join(self.img_dir, item['image']))
                if os.path.exists(image_path):
                    self.valid_data.append((image_path, item['caption']))
                else:
                    logger.warning(f"Файл не найден: {image_path}")
        logger.info(f"Используется {len(self.valid_data)}/{len(self.data)} валидных записей")
    def __len__(self):
        return len(self.valid_data)
    def __getitem__(self, idx):
        while True:
            try:
                image_path, caption = self.valid_data[idx]
                
                # проверка существования файла
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Изображение не найдено: {image_path}")
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    logger.error(f"Ошибка загрузки {image_path}: {str(e)}")
                    raise
                caption = caption.strip()
                if not caption:
                    raise ValueError(f"Пустая аннотация для {image_path}")
                # обработка данных
                inputs = self.processor(
                    images=image,
                    text=caption,
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True)   
                # подготовка данных для модели
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs['labels'] = inputs['input_ids'].clone()
                return inputs
            except Exception:
                idx = random.randint(0, len(self.valid_data)-1)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
dataset = ImageCaptionDataset(JSON_PATH, IMG_DIR, processor)
# конфигурация обучения
training_args = TrainingArguments(
    output_dir="./blip-finetuned",  # папка, куда будут сохраняться чекпойнты модели и другие результаты обучения
    per_device_train_batch_size=1,  # размер батча на одно устройство
    gradient_accumulation_steps=4,  # количество шагов, в течение которых будет накапливаться градиент, эквивалентно увеличению общего размера батча (1 * 4 = 4)
    num_train_epochs=3,  # количество эпох обучения
    learning_rate=3e-5,  # начальное значение скорости обучения, влияет на размер шага при обновлении весов
    warmup_steps=500,  # количество шагов разогрева, когда learning rate постепенно увеличивается от 0 до заданного значения
    weight_decay=0.01,  # коэффициент регуляризации (снижения весов), помогает предотвратить переобучение
    logging_dir="./logs",  # папка для сохранения логов обучения
    logging_steps=1,  # частота логирования, каждую итерацию (1 шаг)
    save_strategy="steps",  # стратегия сохранения модели, по количеству шагов
    save_steps=50,  # сохранять модель каждые 50 шагов
    fp16=False,  # вычисления для ускорения и снижения потребления памяти (False = не использовать)
    report_to="none",  # не отправлять метрики в сторонние сервисы логирования (TensorBoard)
    dataloader_num_workers=0  # количество потоков, используемых для загрузки данных. 0 загрузка будет идти в основном потоке
)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error("Ошибка памяти. Batch_size")
                raise
            logger.error(f"Ошибка при вычислении лосса: {str(e)}")
            raise
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to('cuda')
trainer = CustomTrainer(model=model,args=training_args,train_dataset=dataset,
    data_collator=lambda data: {"pixel_values": torch.stack([item["pixel_values"] for item in data]),"input_ids": torch.stack([item["input_ids"] for item in data]),"attention_mask": torch.stack([item["attention_mask"] for item in data]),
        "labels": torch.stack([item["labels"] for item in data])
    }
)
try:
    logger.info("Начало обучения...")
    trainer.train()
except Exception as e:
    logger.error("Обучение прервано:", exc_info=True)
finally:    
    logger.info("Сохранение модели...")
    model.save_pretrained("./blip-finetuned")
    processor.save_pretrained("./blip-finetuned")
    logger.info("Модель сохранена в ./blip-finetuned")