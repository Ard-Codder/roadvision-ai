@echo off
chcp 65001 >nul
echo.
echo 🚗 RoadVision AI - Обработка всех видео
echo ================================================
echo.

echo 📁 Проверка видео в папке videos/input/...
if not exist "videos\input\*.mp4" (
    echo ❌ Видео файлы не найдены в videos/input/
    echo Поместите видео файлы в папку videos/input/ и запустите батник снова
    pause
    exit /b 1
)

echo ✅ Найдены видео файлы:
dir /b videos\input\*.mp4

echo.
echo 🔄 Начинаем обработку всех видео...
echo.

python process_videos.py --process-all

echo.
echo ✅ Обработка завершена!
echo 📁 Результаты сохранены в videos/output/
echo.
echo Нажмите любую клавишу для выхода...
pause >nul 