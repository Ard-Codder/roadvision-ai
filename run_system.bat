@echo off
chcp 65001 >nul
echo 🚗 RoadVision AI - Система детекции дороги и объектов
echo ==========================================
echo.

echo Выберите действие:
echo 1. Тестирование системы
echo 2. Запуск интерактивного режима
echo 3. Обработка видео (интерактивный)
echo 4. Обработка ВСЕХ видео
echo 5. Установка зависимостей
echo 6. Выход
echo.

set /p choice="Введите номер (1-6): "

if "%choice%"=="1" (
    echo.
    echo 🧪 Запуск тестирования RoadVision AI...
    python test_system.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Запуск RoadVision AI...
    python run_road_detection.py
) else if "%choice%"=="3" (
    echo.
    echo 🎬 Запуск обработки видео (интерактивный)...
    python process_videos.py
) else if "%choice%"=="4" (
    echo.
    echo 🎬 Обработка ВСЕХ видео в папке input...
    call process_all_videos.bat
) else if "%choice%"=="5" (
    echo.
    echo 📦 Установка зависимостей RoadVision AI...
    pip install -r requirements.txt
    echo ✅ Зависимости установлены!
    pause
) else if "%choice%"=="6" (
    echo 👋 До свидания!
    exit /b
) else (
    echo ❌ Неверный выбор!
    pause
) 