#!/usr/bin/env python3
"""
Скрипт для запуска интегрированной системы детекции дороги и объектов
"""

import sys
import os
from integrated_road_detection import RoadDetectionSystem
import cv2

def main():
    """Основная функция запуска"""
    print("🚗 Система детекции дороги и объектов")
    print("=" * 50)
    
    # Инициализация системы
    try:
        system = RoadDetectionSystem()
        print("✅ Система инициализирована успешно!")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # Интерактивное меню
    while True:
        print("\nВыберите действие:")
        print("1. Обработать изображение")
        print("2. Обработать видео")
        print("3. Выход")
        
        choice = input("\nВведите номер (1-3): ").strip()
        
        if choice == "1":
            process_image(system)
        elif choice == "2":
            process_video(system)
        elif choice == "3":
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор. Попробуйте снова.")

def process_image(system):
    """Обработка изображения"""
    print("\n📸 Обработка изображения")
    image_path = input("Введите путь к изображению: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ Файл не найден!")
        return
    
    output_path = input("Введите путь для сохранения результата (или Enter для показа): ").strip()
    if not output_path:
        output_path = None
    
    try:
        system.process_image(image_path, output_path)
        print("✅ Обработка завершена!")
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")

def process_video(system):
    """Обработка видео"""
    print("\n🎬 Обработка видео")
    video_path = input("Введите путь к видео: ").strip()
    
    if not os.path.exists(video_path):
        print("❌ Файл не найден!")
        return
    
    output_path = input("Введите путь для сохранения результата (или Enter для показа): ").strip()
    if not output_path:
        output_path = None
    
    try:
        system.process_video(video_path, output_path)
        print("✅ Обработка завершена!")
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")

if __name__ == "__main__":
    main() 