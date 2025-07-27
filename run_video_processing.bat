@echo off
chcp 65001 >nul
echo.
echo 🚗 RoadVision AI - Обработка видео
echo ====================================
echo.

echo Выберите действие:
echo 1. Показать список видео
echo 2. Обработать все видео
echo 3. Обработать конкретное видео
echo 4. Интерактивный режим
echo 5. Назад в главное меню
echo.

set /p choice="Введите номер (1-5): "

if "%choice%"=="1" (
    echo.
    echo 📋 Список видео в папке input:
    python process_videos.py --list
    pause
) else if "%choice%"=="2" (
    echo.
    echo 🔄 Обработка всех видео...
    python process_videos.py --process-all
    pause
) else if "%choice%"=="3" (
    echo.
    set /p video_name="Введите название видео файла: "
    echo 🔄 Обработка видео %video_name%...
    python process_videos.py --process %video_name%
    pause
) else if "%choice%"=="4" (
    echo.
    echo 🎬 Запуск интерактивного режима...
    python process_videos.py
) else if "%choice%"=="5" (
    echo.
    call run_system.bat
) else (
    echo ❌ Неверный выбор!
    pause
) 