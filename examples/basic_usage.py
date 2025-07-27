#!/usr/bin/env python3
"""
Пример базового использования Road Detection System
"""

import cv2
import numpy as np
from integrated_road_detection import RoadDetectionSystem

def main():
    """Пример использования системы"""
    print("🚗 Пример использования Road Detection System")
    print("=" * 50)
    
    # Инициализация системы
    print("1. Инициализация системы...")
    system = RoadDetectionSystem()
    print("✅ Система готова к работе!")
    
    # Создание тестового изображения
    print("\n2. Создание тестового изображения...")
    test_image = create_test_image()
    cv2.imwrite("test_input.jpg", test_image)
    print("✅ Тестовое изображение создано: test_input.jpg")
    
    # Обработка изображения
    print("\n3. Обработка изображения...")
    processed_image = system.process_frame(test_image)
    cv2.imwrite("test_output.jpg", processed_image)
    print("✅ Результат сохранен: test_output.jpg")
    
    # Обработка видео (если есть)
    print("\n4. Проверка видео файлов...")
    import os
    video_files = [f for f in os.listdir("videos/input") if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if video_files:
        print(f"Найдено видео файлов: {len(video_files)}")
        print("Для обработки видео используйте:")
        print("python process_videos.py --list")
        print("python process_videos.py --process-all")
    else:
        print("Видео файлы не найдены в videos/input/")
        print("Поместите видео файлы в папку videos/input/ для обработки")
    
    print("\n🎉 Пример завершен!")
    print("Проверьте созданные файлы:")
    print("- test_input.jpg - исходное изображение")
    print("- test_output.jpg - обработанное изображение")

def create_test_image():
    """Создание тестового изображения с дорогой и объектами"""
    # Создаем изображение 640x480
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Рисуем дорогу (серый цвет)
    cv2.rectangle(image, (0, 200), (640, 480), (128, 128, 128), -1)
    
    # Рисуем машину (красный прямоугольник)
    cv2.rectangle(image, (100, 300), (200, 400), (0, 0, 255), -1)
    cv2.putText(image, "car", (100, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Рисуем человека (зеленый круг)
    cv2.circle(image, (400, 350), 30, (0, 255, 0), -1)
    cv2.putText(image, "person", (350, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Рисуем знак (красный треугольник)
    pts = np.array([[500, 200], [480, 250], [520, 250]], np.int32)
    cv2.fillPoly(image, [pts], (0, 0, 255))
    cv2.putText(image, "stop sign", (470, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Рисуем дерево (зеленый круг)
    cv2.circle(image, (150, 150), 25, (0, 128, 0), -1)
    cv2.putText(image, "tree", (130, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
    
    return image

if __name__ == "__main__":
    main() 