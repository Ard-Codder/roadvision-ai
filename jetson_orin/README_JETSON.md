# 🚗 Road Detection System для NVIDIA Jetson Orin

Оптимизированная версия системы детекции дороги и объектов для NVIDIA Jetson Orin с поддержкой камеры и GPU ускорением.

## 🎯 Особенности для Jetson Orin

### 🚀 Оптимизации
- **GPU ускорение** - использование CUDA для PyTorch и OpenVINO
- **Многопоточность** - отдельные потоки для захвата и обработки кадров
- **Оптимизированные модели** - YOLOv8n-seg для быстрой детекции
- **Реальное время** - обработка в реальном времени с FPS мониторингом

### 📹 Поддержка камеры
- **Автоматическое определение** камеры
- **Настраиваемые параметры** (разрешение, FPS)
- **Многопоточный захват** кадров
- **Сохранение кадров** по нажатию клавиши

### 🎮 Управление
- **'q'** - выход из программы
- **'s'** - сохранение текущего кадра

## 📦 Установка на Jetson Orin

### 1. Подготовка системы
```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка необходимых пакетов
sudo apt install -y python3-pip python3-dev
sudo apt install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### 2. Установка PyTorch для Jetson
```bash
# Установка PyTorch с поддержкой CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Установка зависимостей
```bash
# Переход в папку jetson_orin
cd jetson_orin

# Установка зависимостей
pip3 install -r requirements_jetson.txt
```

### 4. Установка дополнительных утилит
```bash
# Мониторинг системы Jetson
sudo pip3 install jetson-stats

# Утилиты NVIDIA
sudo pip3 install nvidia-ml-py
```

## 🚀 Запуск

### Базовый запуск
```bash
python3 road_detection_jetson.py
```

### С дополнительными параметрами
```bash
# Использование другой камеры
python3 road_detection_jetson.py --camera 1

# Только CPU (без GPU)
python3 road_detection_jetson.py --cpu-only

# Комбинация параметров
python3 road_detection_jetson.py --camera 0 --cpu-only
```

## 🔧 Настройка камеры

### Проверка доступных камер
```bash
# Список доступных камер
ls /dev/video*

# Тест камеры
v4l2-ctl --list-devices
```

### Настройка параметров камеры
В файле `road_detection_jetson.py` можно изменить параметры камеры:

```python
# В методе setup_camera()
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Ширина
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    # Высота
self.cap.set(cv2.CAP_PROP_FPS, 30)              # FPS
```

## 📊 Мониторинг производительности

### Системные утилиты
```bash
# Мониторинг GPU
nvidia-smi

# Мониторинг системы Jetson
jtop

# Мониторинг температуры
cat /sys/class/thermal/thermal_zone*/temp
```

### Встроенный мониторинг
- **FPS** отображается на экране
- **Статистика объектов** в реальном времени
- **Автоматическое сохранение** кадров

## 🎯 Оптимизация производительности

### Настройка мощности Jetson
```bash
# Максимальная производительность
sudo nvpmodel -m 0

# Сбалансированный режим
sudo nvpmodel -m 1

# Энергосберегающий режим
sudo nvpmodel -m 2
```

### Настройка вентилятора
```bash
# Автоматическое управление
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'

# Ручное управление (0-255)
sudo sh -c 'echo 128 > /sys/devices/pwm-fan/target_pwm'
```

## 🔍 Устранение неполадок

### Проблемы с камерой
```bash
# Проверка камеры
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Тест камеры
ffplay /dev/video0
```

### Проблемы с GPU
```bash
# Проверка CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Проверка OpenVINO
python3 -c "import openvino as ov; print(ov.Core().get_available_devices())"
```

### Проблемы с памятью
```bash
# Мониторинг памяти
free -h

# Очистка кэша
sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches
```

## 📁 Структура файлов

```
jetson_orin/
├── road_detection_jetson.py    # Основной файл системы
├── requirements_jetson.txt      # Зависимости для Jetson
├── README_JETSON.md            # Документация
└── model/                      # Папка для моделей
    ├── road-segmentation-adas-0001.xml
    └── road-segmentation-adas-0001.bin
```

## 🎮 Использование

### Запуск системы
1. Подключите камеру к Jetson Orin
2. Запустите систему: `python3 road_detection_jetson.py`
3. Нажмите 'q' для выхода, 's' для сохранения кадра

### Результаты
- **Реальное время** - обработка кадров в реальном времени
- **FPS мониторинг** - отображение производительности
- **Сохранение кадров** - автоматическое сохранение при нажатии 's'
- **Статистика объектов** - подсчет детектированных объектов

## 🔧 Технические детали

### Многопоточность
- **Поток камеры** - захват кадров с камеры
- **Поток обработки** - обработка кадров с моделями
- **Основной поток** - отображение результатов

### Оптимизации памяти
- **Очереди кадров** - ограниченный размер для предотвращения утечек
- **Автоматическая очистка** - удаление старых кадров
- **GPU память** - оптимизированное использование CUDA

### Поддерживаемые камеры
- **USB камеры** - стандартные USB камеры
- **CSI камеры** - встроенные камеры Jetson
- **IP камеры** - сетевые камеры (с дополнительной настройкой)

## 📞 Поддержка

При возникновении проблем:
1. Проверьте подключение камеры
2. Убедитесь в установке всех зависимостей
3. Проверьте доступность GPU: `nvidia-smi`
4. Проверьте температуру системы: `jtop`

## 📄 Лицензия

MIT License - см. файл LICENSE в корне проекта. 