#!/usr/bin/env python3
"""
Road Detection System для NVIDIA Jetson Orin
Оптимизированная версия с поддержкой камеры и GPU ускорением
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import openvino as ov
from pathlib import Path
import time
import os
import requests
import threading
import queue
from typing import Optional, Tuple

class JetsonRoadDetectionSystem:
    def __init__(self, camera_index: int = 0, gpu_acceleration: bool = True):
        """
        Инициализация системы для Jetson Orin
        
        Args:
            camera_index: Индекс камеры (0 для основной)
            gpu_acceleration: Использовать GPU ускорение
        """
        self.camera_index = camera_index
        self.gpu_acceleration = gpu_acceleration
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Инициализация моделей
        self.setup_models()
        self.setup_colors()
        
        # Настройка камеры
        self.setup_camera()
        
        print("🚗 Система детекции дороги для Jetson Orin готова!")
    
    def setup_models(self):
        """Инициализация моделей с оптимизацией для Jetson"""
        print("Загрузка моделей для Jetson Orin...")
        
        # Проверка доступности GPU
        if self.gpu_acceleration and torch.cuda.is_available():
            device = "cuda"
            print(f"✅ GPU доступен: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("⚠️ GPU недоступен, используем CPU")
        
        # YOLOv8 для детекции объектов (оптимизированная модель)
        self.yolo_model = YOLO('yolov8n-seg.pt')
        self.yolo_model.to(device)
        
        # OpenVINO для сегментации дороги
        self.setup_openvino_road_model()
        
        print("✅ Модели загружены успешно!")
    
    def download_file(self, url: str, filename: str, directory: str) -> Optional[str]:
        """Скачивание файлов с проверкой"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        if os.path.exists(filepath):
            print(f"Файл {filename} уже существует")
            return filepath
        
        print(f"Скачивание {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ {filename} скачан успешно")
            return filepath
        except Exception as e:
            print(f"❌ Ошибка скачивания {filename}: {e}")
            return None
    
    def setup_openvino_road_model(self):
        """Настройка модели сегментации дороги OpenVINO"""
        base_model_dir = Path("./model").expanduser()
        base_model_dir.mkdir(exist_ok=True)
        
        model_name = "road-segmentation-adas-0001"
        model_xml_name = f'{model_name}.xml'
        model_bin_name = f'{model_name}.bin'
        
        model_xml_path = base_model_dir / model_xml_name
        
        if not model_xml_path.exists():
            print("Скачивание модели сегментации дороги...")
            model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml"
            model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin"
            
            self.download_file(model_xml_url, model_xml_name, base_model_dir)
            self.download_file(model_bin_url, model_bin_name, base_model_dir)
        
        # Загрузка модели OpenVINO с оптимизацией для Jetson
        core = ov.Core()
        self.road_model = core.read_model(model=model_xml_path)
        
        # Используем GPU если доступен
        if self.gpu_acceleration:
            try:
                self.compiled_road_model = core.compile_model(model=self.road_model, device_name="GPU")
                print("✅ OpenVINO модель загружена на GPU")
            except:
                self.compiled_road_model = core.compile_model(model=self.road_model, device_name="CPU")
                print("⚠️ OpenVINO модель загружена на CPU")
        else:
            self.compiled_road_model = core.compile_model(model=self.road_model, device_name="CPU")
        
        self.road_input_layer = self.compiled_road_model.input(0)
        self.road_output_layer = self.compiled_road_model.output(0)
    
    def setup_colors(self):
        """Настройка цветов для визуализации"""
        # Цветовая схема для объектов улицы
        self.object_colors = {
            # Транспорт
            'car': (255, 0, 0),         # 🔵 Синий
            'truck': (255, 0, 0),       # 🔵 Синий
            'bus': (255, 0, 0),         # 🔵 Синий
            'motorcycle': (255, 0, 0),  # 🔵 Синий
            'bicycle': (255, 0, 0),     # 🔵 Синий
            'train': (255, 0, 0),       # 🔵 Синий
            'airplane': (255, 0, 0),    # 🔵 Синий
            'boat': (255, 0, 0),        # 🔵 Синий
            
            # Люди
            'person': (0, 255, 0),      # 🟢 Зеленый
            
            # Дорожная инфраструктура
            'stop sign': (0, 0, 255),   # 🔴 Красный
            'traffic light': (0, 255, 255), # 🟡 Желтый
            'fire hydrant': (255, 255, 0),  # 🔵 Голубой
            'parking meter': (0, 255, 255), # 🟡 Желтый
            'bench': (128, 0, 128),     # 🟣 Фиолетовый
            
            # Дорожные объекты
            'pole': (255, 165, 0),      # 🟠 Оранжевый
            'street light': (255, 255, 0), # 🟡 Желтый
            'construction': (139, 69, 19), # 🟤 Коричневый
            'fence': (128, 128, 128),   # ⚪ Серый
            
            # Природа
            'tree': (34, 139, 34),      # 🌳 Темно-зеленый
            'plant': (0, 128, 0),       # 🌿 Зеленый
            'grass': (0, 255, 127),     # 🌱 Светло-зеленый
            
            # Здания и сооружения
            'building': (105, 105, 105), # 🏢 Темно-серый
            'house': (139, 69, 19),     # 🏠 Коричневый
            'bridge': (160, 82, 45),    # 🌉 Коричневый
            'tunnel': (47, 79, 79),     # 🚇 Темно-серый
            
            # Животные
            'dog': (255, 20, 147),      # 🐕 Розовый
            'cat': (255, 20, 147),      # 🐱 Розовый
            'horse': (255, 20, 147),    # 🐎 Розовый
            'sheep': (255, 20, 147),    # 🐑 Розовый
            'cow': (255, 20, 147),      # 🐄 Розовый
            'elephant': (255, 20, 147), # 🐘 Розовый
            'bear': (255, 20, 147),     # 🐻 Розовый
            'zebra': (255, 20, 147),    # 🦓 Розовый
            'giraffe': (255, 20, 147),  # 🦒 Розовый
        }
        
        # Цвета для сегментации дороги
        self.road_colors = {
            0: [68, 1, 84],    # Фон - темно-фиолетовый
            1: [48, 103, 141], # Дорога - синий
            2: [53, 183, 120], # Бордюр - зеленый
            3: [199, 216, 52]  # Разметка - желтый
        }
    
    def setup_camera(self):
        """Настройка камеры для Jetson"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Настройка параметров камеры для лучшей производительности
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print(f"❌ Ошибка открытия камеры {self.camera_index}")
            raise RuntimeError("Камера недоступна")
        
        print(f"✅ Камера {self.camera_index} инициализирована")
    
    def segment_road(self, image: np.ndarray) -> np.ndarray:
        """Сегментация дороги с помощью OpenVINO модели"""
        N, C, H, W = self.road_input_layer.shape
        resized_image = cv2.resize(image, (W, H))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        
        # Инференс
        result = self.compiled_road_model([input_image])[self.road_output_layer]
        segmentation_mask = np.argmax(result, axis=1)
        
        # Преобразование маски в цветное изображение
        mask = self.segmentation_map_to_image(segmentation_mask, self.road_colors)
        resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        return resized_mask
    
    def segmentation_map_to_image(self, segmentation_mask: np.ndarray, colormap: dict) -> np.ndarray:
        """Преобразование маски сегментации в цветное изображение"""
        mask = np.zeros((segmentation_mask.shape[1], segmentation_mask.shape[2], 3), dtype=np.uint8)
        
        for class_id, color in colormap.items():
            mask[segmentation_mask[0] == class_id] = color
        
        return mask
    
    def detect_objects(self, image: np.ndarray):
        """Детекция объектов с помощью YOLOv8"""
        results = self.yolo_model(image, verbose=False)
        return results[0]
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Обработка одного кадра"""
        # Сегментация дороги
        road_mask = self.segment_road(frame)
        
        # Детекция объектов
        results = self.detect_objects(frame)
        
        # Создание результата
        result_frame = frame.copy()
        
        # Наложение маски дороги с прозрачностью
        alpha = 0.3
        result_frame = cv2.addWeighted(road_mask, alpha, result_frame, 1 - alpha, 0)
        
        # Счетчики объектов
        object_counts = {}
        
        # Отрисовка объектов с сегментацией
        if hasattr(results, 'masks') and results.masks is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = results.names[cls]
                
                # Проверяем, что объект входит в нашу цветовую схему
                if class_name in self.object_colors and conf > 0.5:
                    # Подсчет объектов
                    if class_name not in object_counts:
                        object_counts[class_name] = 0
                    object_counts[class_name] += 1
                    
                    color = self.object_colors[class_name]
                    
                    # Отрисовка bbox
                    cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(result_frame, f'{class_name} {conf:.2f}', 
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Отрисовка маски сегментации объекта
                    if i < len(results.masks) and results.masks[i] is not None:
                        try:
                            # Получаем данные маски правильным способом
                            mask_data = results.masks[i].data.cpu().numpy()
                            if mask_data is not None and mask_data.size > 0:
                                # Изменяем размер маски под размер кадра
                                mask_resized = cv2.resize(mask_data[0], (frame.shape[1], frame.shape[0]))
                                
                                # Создаем маску объекта
                                mask_bool = mask_resized > 0.5
                                
                                # Накладываем маску напрямую без изменения яркости
                                # Создаем цветную маску
                                colored_mask = np.zeros_like(frame)
                                colored_mask[mask_bool] = color
                                
                                # Накладываем маску с прозрачностью, но без затемнения
                                mask_alpha = 0.4
                                # Используем numpy операции вместо cv2.addWeighted
                                result_frame[mask_bool] = (
                                    result_frame[mask_bool] * (1 - mask_alpha) + 
                                    colored_mask[mask_bool] * mask_alpha
                                ).astype(np.uint8)
                                
                        except Exception as e:
                            print(f"Ошибка обработки маски для {class_name}: {e}")
        else:
            # Fallback для моделей без сегментации
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = results.names[cls]
                
                # Проверяем, что объект входит в нашу цветовую схему
                if class_name in self.object_colors and conf > 0.5:
                    if class_name not in object_counts:
                        object_counts[class_name] = 0
                    object_counts[class_name] += 1
                    
                    color = self.object_colors[class_name]
                    cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(result_frame, f'{class_name} {conf:.2f}', 
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Отображение статистики объектов
        if object_counts:
            stats_text = "Objects: " + ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])
            cv2.putText(result_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Добавляем информацию о производительности
        if hasattr(self, 'fps'):
            cv2.putText(result_frame, f"FPS: {self.fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame
    
    def camera_thread(self):
        """Поток для захвата кадров с камеры"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Ошибка чтения кадра с камеры")
                break
            
            # Очищаем очередь если она полная
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
    
    def processing_thread(self):
        """Поток для обработки кадров"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Обработка кадра
            processed_frame = self.process_frame(frame)
            
            # Обновление FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                self.fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Очищаем очередь результатов если она полная
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.result_queue.put(processed_frame)
    
    def run_realtime(self):
        """Запуск системы в реальном времени"""
        print("🚀 Запуск системы в реальном времени...")
        print("Нажмите 'q' для выхода, 's' для сохранения кадра")
        
        self.running = True
        
        # Запуск потоков
        camera_thread = threading.Thread(target=self.camera_thread)
        processing_thread = threading.Thread(target=self.processing_thread)
        
        camera_thread.start()
        processing_thread.start()
        
        frame_count = 0
        
        try:
            while True:
                try:
                    processed_frame = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Показ результата
                cv2.imshow('Jetson Road Detection System', processed_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Сохранение кадра
                    filename = f"jetson_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"💾 Кадр сохранен: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹️ Остановка системы...")
        
        finally:
            self.running = False
            camera_thread.join()
            processing_thread.join()
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Система остановлена")

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Road Detection System для Jetson Orin")
    parser.add_argument("--camera", type=int, default=0, help="Индекс камеры (по умолчанию 0)")
    parser.add_argument("--cpu-only", action="store_true", help="Использовать только CPU")
    
    args = parser.parse_args()
    
    try:
        # Создание системы
        system = JetsonRoadDetectionSystem(
            camera_index=args.camera,
            gpu_acceleration=not args.cpu_only
        )
        
        # Запуск в реальном времени
        system.run_realtime()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 