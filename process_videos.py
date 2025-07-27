#!/usr/bin/env python3
"""
Скрипт для обработки видео с сегментацией дороги и детекцией объектов
"""

import os
import cv2
import time
import argparse
from pathlib import Path
from integrated_road_detection import RoadDetectionSystem

class VideoProcessor:
    def __init__(self):
        """Инициализация процессора видео"""
        print("🚗 Инициализация системы обработки видео...")
        self.system = RoadDetectionSystem()
        print("✅ Система готова к обработке видео!")
    
    def get_video_files(self, input_dir="videos/input"):
        """Получение списка видео файлов"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Папка {input_dir} не найдена!")
            return []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = []
        
        for file in input_path.iterdir():
            if file.suffix.lower() in video_extensions:
                video_files.append(file)
        
        return video_files
    
    def process_single_video(self, video_path, output_dir="videos/output", show_preview=False):
        """Обработка одного видео"""
        print(f"\n🎬 Обработка видео: {video_path.name}")
        
        # Создание папки для результатов
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Путь для сохранения результата
        output_file = output_path / f"processed_{video_path.name}"
        
        # Открытие видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео: {video_path}")
            return False
        
        # Получение параметров видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Параметры видео:")
        print(f"   - Разрешение: {width}x{height}")
        print(f"   - FPS: {fps}")
        print(f"   - Всего кадров: {total_frames}")
        
        # Настройка записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        # Обработка кадров
        frame_count = 0
        start_time = time.time()
        
        print("🔄 Обработка кадров...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обработка кадра
            processed_frame = self.system.process_frame(frame)
            
            # Запись результата
            out.write(processed_frame)
            
            # Показ превью
            if show_preview:
                cv2.imshow('Video Processing', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Прогресс каждые 30 кадров
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"   Прогресс: {progress:.1f}% | Кадров: {frame_count}/{total_frames} | FPS: {current_fps:.1f}")
        
        # Завершение
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"✅ Обработка завершена!")
        print(f"   - Время: {total_time:.1f} сек")
        print(f"   - Средний FPS: {avg_fps:.1f}")
        print(f"   - Результат: {output_file}")
        
        return True
    
    def process_all_videos(self, input_dir="videos/input", output_dir="videos/output", show_preview=False):
        """Обработка всех видео в папке"""
        video_files = self.get_video_files(input_dir)
        
        if not video_files:
            print(f"❌ Видео файлы не найдены в папке {input_dir}")
            print("📁 Поместите видео файлы в папку videos/input/")
            return
        
        print(f"📹 Найдено видео файлов: {len(video_files)}")
        for i, video_file in enumerate(video_files, 1):
            print(f"   {i}. {video_file.name}")
        
        print(f"\n🚀 Начинаем обработку {len(video_files)} видео...")
        
        successful = 0
        failed = 0
        
        for video_file in video_files:
            try:
                if self.process_single_video(video_file, output_dir, show_preview):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Ошибка обработки {video_file.name}: {e}")
                failed += 1
        
        print(f"\n📊 ИТОГОВЫЙ ОТЧЕТ")
        print(f"   ✅ Успешно обработано: {successful}")
        print(f"   ❌ Ошибок: {failed}")
        print(f"   📁 Результаты сохранены в: {output_dir}")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Обработка видео с сегментацией дороги")
    parser.add_argument("--list", action="store_true", help="Показать список видео файлов")
    parser.add_argument("--process-all", action="store_true", help="Обработать все видео")
    parser.add_argument("--process", type=str, help="Обработать конкретное видео")
    parser.add_argument("--input-dir", default="videos/input", help="Папка с входными видео")
    parser.add_argument("--output-dir", default="videos/output", help="Папка для результатов")
    parser.add_argument("--show-preview", action="store_true", help="Показать превью во время обработки")
    
    args = parser.parse_args()
    
    processor = VideoProcessor()
    
    if args.list:
        # Показать список видео
        video_files = processor.get_video_files(args.input_dir)
        if video_files:
            print(f"📹 Найдено видео файлов в {args.input_dir}:")
            for i, video_file in enumerate(video_files, 1):
                print(f"   {i}. {video_file.name}")
        else:
            print(f"❌ Видео файлы не найдены в {args.input_dir}")
    
    elif args.process_all:
        # Обработать все видео
        processor.process_all_videos(args.input_dir, args.output_dir, args.show_preview)
    
    elif args.process:
        # Обработать конкретное видео
        video_path = Path(args.input_dir) / args.process
        if video_path.exists():
            processor.process_single_video(video_path, args.output_dir, args.show_preview)
        else:
            print(f"❌ Видео файл не найден: {video_path}")
    
    else:
        # Интерактивный режим
        print("🚗 Система обработки видео")
        print("=" * 50)
        
        while True:
            print("\nВыберите действие:")
            print("1. Показать список видео")
            print("2. Обработать все видео")
            print("3. Обработать конкретное видео")
            print("4. Выход")
            
            choice = input("\nВведите номер (1-4): ").strip()
            
            if choice == "1":
                video_files = processor.get_video_files()
                if video_files:
                    print(f"\n📹 Найдено видео файлов: {len(video_files)}")
                    for i, video_file in enumerate(video_files, 1):
                        print(f"   {i}. {video_file.name}")
                else:
                    print("❌ Видео файлы не найдены в videos/input/")
            
            elif choice == "2":
                show_preview = input("Показать превью во время обработки? (y/n): ").lower() == 'y'
                processor.process_all_videos(show_preview=show_preview)
            
            elif choice == "3":
                video_files = processor.get_video_files()
                if video_files:
                    print("\nДоступные видео:")
                    for i, video_file in enumerate(video_files, 1):
                        print(f"   {i}. {video_file.name}")
                    
                    try:
                        video_choice = int(input("\nВведите номер видео: ")) - 1
                        if 0 <= video_choice < len(video_files):
                            show_preview = input("Показать превью? (y/n): ").lower() == 'y'
                            processor.process_single_video(video_files[video_choice], show_preview=show_preview)
                        else:
                            print("❌ Неверный номер!")
                    except ValueError:
                        print("❌ Введите число!")
                else:
                    print("❌ Видео файлы не найдены!")
            
            elif choice == "4":
                print("👋 До свидания!")
                break
            
            else:
                print("❌ Неверный выбор!")

if __name__ == "__main__":
    main() 