# 🤝 Руководство по вкладу в RoadVision AI

Спасибо за интерес к проекту RoadVision AI! Мы приветствуем вклады от сообщества.

## 🚀 Быстрый старт

1. **Форкните репозиторий**
   ```bash
   git clone https://github.com/Ard-Codder/roadvision-ai.git
   cd roadvision-ai
   ```

2. **Создайте ветку для новой функции**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Внесите изменения и зафиксируйте их**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

4. **Отправьте изменения в ваш форк**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Создайте Pull Request**

## 📋 Стандарты кода

### Python
- Используйте **Python 3.8+**
- Следуйте **PEP 8** для стиля кода
- Используйте **type hints** где возможно
- Добавляйте **docstrings** для функций и классов

### Пример хорошего кода:
```python
def process_frame(frame: np.ndarray, confidence: float = 0.5) -> np.ndarray:
    """
    Обрабатывает кадр видео для детекции объектов.
    
    Args:
        frame: Входной кадр
        confidence: Порог уверенности для детекции
        
    Returns:
        Обработанный кадр с детекциями
    """
    # Ваш код здесь
    return processed_frame
```

### Коммиты
Используйте [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - новая функциональность
- `fix:` - исправление бага
- `docs:` - изменения в документации
- `style:` - форматирование кода
- `refactor:` - рефакторинг кода
- `test:` - добавление тестов
- `chore:` - обновление зависимостей

Примеры:
```bash
git commit -m "feat: add support for new object classes"
git commit -m "fix: resolve video darkening issue"
git commit -m "docs: update installation instructions"
```

## 🧪 Тестирование

### Запуск тестов
```bash
# Все тесты
python -m pytest

# Конкретный тест
python -m pytest tests/test_detection.py

# С покрытием
python -m pytest --cov=roadvision_ai
```

### Написание тестов
```python
import pytest
import numpy as np
from roadvision_ai import RoadDetectionSystem

def test_object_detection():
    """Тест детекции объектов"""
    system = RoadDetectionSystem()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = system.process_frame(test_image)
    
    assert result is not None
    assert result.shape == test_image.shape
```

## 📚 Документация

### Обновление README
- Добавляйте новые возможности в соответствующие секции
- Обновляйте примеры использования
- Добавляйте скриншоты для новых функций

### Документирование кода
```python
class RoadDetectionSystem:
    """
    Основная система для детекции дороги и объектов.
    
    Attributes:
        model_path (str): Путь к модели YOLOv8
        confidence (float): Порог уверенности
        device (str): Устройство для инференса (cpu/cuda)
    """
    
    def __init__(self, model_path: str = "model/yolov8n-seg.pt"):
        """
        Инициализирует систему детекции.
        
        Args:
            model_path: Путь к файлу модели
        """
        self.model_path = model_path
        # Инициализация
```

## 🐛 Сообщение о багах

### Создание Issue
1. Проверьте существующие issues
2. Создайте новый issue с четким описанием
3. Включите:
   - Описание проблемы
   - Шаги для воспроизведения
   - Ожидаемое поведение
   - Версии зависимостей
   - Скриншоты (если применимо)

### Шаблон для бага:
```markdown
**Описание бага**
Краткое описание проблемы.

**Шаги для воспроизведения**
1. Запустите '...'
2. Нажмите '....'
3. Прокрутите до '....'
4. Увидите ошибку

**Ожидаемое поведение**
Что должно происходить.

**Окружение**
- OS: Windows 10
- Python: 3.9.7
- OpenCV: 4.8.0
- PyTorch: 2.0.0

**Дополнительная информация**
Любая дополнительная информация.
```

## 💡 Запросы функций

### Создание Feature Request
1. Опишите проблему, которую решает функция
2. Предложите решение
3. Рассмотрите альтернативы
4. Добавьте контекст

### Шаблон для запроса функции:
```markdown
**Проблема**
Описание проблемы, которую решает функция.

**Предлагаемое решение**
Описание желаемого решения.

**Альтернативы**
Другие возможные решения.

**Дополнительный контекст**
Любая дополнительная информация.
```

## 🔧 Настройка разработки

### Установка зависимостей разработки
```bash
# Клонирование
git clone https://github.com/Ard-Codder/roadvision-ai.git
cd roadvision-ai

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
pip install -e .

# Установка инструментов разработки
pip install pytest black flake8 mypy
```

### Инструменты разработки
```bash
# Форматирование кода
black integrated_road_detection.py

# Проверка стиля
flake8 integrated_road_detection.py

# Проверка типов
mypy integrated_road_detection.py

# Запуск тестов
pytest
```

## 📦 Релизы

### Создание релиза
1. Обновите версию в `setup.py`
2. Добавьте изменения в `docs/project/CHANGELOG.md`
3. Создайте тег:
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```

### Версионирование
Следуем [Semantic Versioning](https://semver.org/):
- **MAJOR** - несовместимые изменения API
- **MINOR** - новая функциональность с обратной совместимостью
- **PATCH** - исправления багов с обратной совместимостью

## 🏷️ Labels для Issues

- `bug` - Ошибки и проблемы
- `enhancement` - Улучшения и новые функции
- `documentation` - Изменения в документации
- `good first issue` - Хорошие задачи для новичков
- `help wanted` - Нужна помощь
- `question` - Вопросы
- `wontfix` - Не будет исправлено

## 📞 Связь

- **Issues**: [GitHub Issues](https://github.com/Ard-Codder/roadvision-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ard-Codder/roadvision-ai/discussions)
- **Email**: contact@roadvision-ai.com

## 🙏 Благодарности

Спасибо всем контрибьюторам, которые помогают сделать RoadVision AI лучше!

---

**Вместе мы создаем будущее компьютерного зрения!** 🚗✨ 