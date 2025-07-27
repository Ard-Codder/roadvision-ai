@echo off
chcp 65001 >nul
echo.
echo 🚗 RoadVision AI - Быстрая обработка всех видео
echo ================================
echo.

echo 📁 Видео в папке input:
dir /b videos\input\*.mp4 2>nul
if errorlevel 1 (
    echo ❌ Видео не найдены в videos/input/
    echo Поместите видео файлы в папку videos/input/
    pause
    exit /b 1
)

echo.
echo 🔄 Начинаем обработку всех видео...
echo.

python process_videos.py --process-all

echo.
echo ✅ Обработка завершена!
echo 📁 Результаты в videos/output/
echo.
pause 