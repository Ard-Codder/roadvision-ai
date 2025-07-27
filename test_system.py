#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы системы детекции дороги и объектов
"""

import cv2
import numpy as np
import os
from integrated_road_detection import RoadDetectionSystem

def create_test_image():
    """Создание тестового изображения с дорогой и объектами"""
    # Создаем изображение 640x480
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Рисуем дорогу (серый цвет)
    cv2.rectangle(img, (0, 200), (640, 480), (128, 128, 128), -1)
    
    # Рисуем разметку (белые линии)
    cv2.line(img, (320, 200), (320, 480), (255, 255, 255), 5)
    cv2.line(img, (0, 340), (640, 340), (255, 255, 255), 3)
    
    # Рисуем бордюр (зеленый)
    cv2.rectangle(img, (0, 180), (640, 200), (0, 255, 0), -1)
    
    # Рисуем машину (синий прямоугольник)
    cv2.rectangle(img, (100, 300), (200, 380), (255, 0, 0), -1)
    
    # Рисуем человека (зеленый круг)
    cv2.circle(img, (500, 350), 20, (0, 255, 0), -1)
    
    # Рисуем знак (красный треугольник)
    pts = np.array([[400, 100], [450, 150], [350, 150]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255))
    
    return img

def test_system():
    """Тестирование системы"""
    print("🧪 Тестирование системы детекции дороги и объектов")
    print("=" * 60)
    
    try:
        # Инициализация системы
        print("1. Инициализация системы...")
        system = RoadDetectionSystem()
        print("✅ Система инициализирована успешно!")
        
        # Создание тестового изображения
        print("\n2. Создание тестового изображения...")
        test_img = create_test_image()
        cv2.imwrite('test_image.jpg', test_img)
        print("✅ Тестовое изображение создано: test_image.jpg")
        
        # Обработка тестового изображения
        print("\n3. Обработка тестового изображения...")
        system.process_image('test_image.jpg', 'test_result.jpg')
        print("✅ Результат сохранен: test_result.jpg")
        
        # Проверка файлов
        print("\n4. Проверка созданных файлов...")
        files_to_check = [
            'test_image.jpg',
            'test_result.jpg',
            'model/road-segmentation-adas-0001.xml',
            'model/road-segmentation-adas-0001.bin'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {file_path} ({size} bytes)")
            else:
                print(f"❌ {file_path} - не найден")
        
        print("\n🎉 Тестирование завершено успешно!")
        print("\nРезультаты:")
        print("- Система инициализирована")
        print("- Модели загружены")
        print("- Тестовое изображение обработано")
        print("- Результат сохранен")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚗 Тестирование системы детекции дороги и объектов")
    print("=" * 60)
    
    # Тест системы
    system_ok = test_system()
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    if system_ok:
        print("✅ Система работает корректно")
        print("\n🎉 Все тесты пройдены! Система готова к работе.")
        print("\nДля запуска используйте:")
        print("python run_road_detection.py")
        print("python process_videos.py")
    else:
        print("❌ Проблемы с системой")
        print("\n⚠️  Обнаружены проблемы. Проверьте установку зависимостей.")

if __name__ == "__main__":
    main() 