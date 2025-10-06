# cvat-tracks-generator

### Вимоги

* Python 3.12+
* PDM (для управління пакетами та середовищем)

### Встановлення (локально через PDM)

Встанови залежності та активуй середовище:

```bash
pdm install
.\.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate      # Linux/macOS
```

### Встановлення як пакет (після публікації)

```bash
pdm build
pip install dist/cvat_tracks_generator-0.1.0-py3-none-any.whl
```

### CLI команди перевірки після встановлення:

```bash
cvat-gen
cvat-gen --help
```

### Запуск тестів

Щоб перевірити роботу пакета та модулів:

```bash
pdm run pytest
```

Або напряму:

```bash
pytest
```

### Приклади команд

1) 
1.1) Детекція+трекінг (ByteTrack) та експорт у XML, з опційною візуалізацією:
```bash
cvat-gen detect-track --model yolov8s-visdrone.pt --video example.mp4 --out-xml tracks.xml --save-video out_vis.mp4
```

1.2) Детекція з SAHI:
```bash
cvat-gen detect-track --model yolov8s-visdrone.pt --video example.mp4 --out-xml tracks_sahi.xml --use-sahi --save-video out_vis_sahi.mp4
```

Параметри SAHI читаються з `.env` з префіксом `SAHI_` 
```bash
SAHI_SLICE_HEIGHT=640
SAHI_SLICE_WIDTH=640
SAHI_OVERLAP_HEIGHT_RATIO=0.2
SAHI_OVERLAP_WIDTH_RATIO=0.2
SAHI_CONF=0.25
SAHI_IOU_THRESHOLD=0.3
SAHI_MAX_AGE=30
```

2) Візуалізація XML на відео:
```bash
cvat-gen render --xml tracks.xml --video example.mp4 --out-video tracks_vis.mp4
```

3) Об'єднання та видалення треків у XML:
```bash
cvat-gen edit --xml tracks.xml --out-xml tracks_edited.xml --merge 3,5 --delete 10,11
```

4) Об'єднання та видалення треків у XML з візуалізацією:
```bash
cvat-gen edit --xml tracks.xml --out-xml tracks_edited.xml --merge 3,5 --delete 10,11 --video example.mp4 --save-video edited_vis.mp4
```

### Формат XML
Проект генерує повний CVAT for video 1.1 XML з усіма метаданими:
- Версія формату `<version>1.1</version>`
- Повна інформація про task (розміри відео, дати, labels)
- Треки з `source="manual"` та `z_order="0"`
- Автоматична генерація кольорів для labels

### Приклад
Опен-сорс модель натренована на датасеті VisDrone: mshamrai/yolov8s-visdrone · Hugging Face 
(Завантажити `.pt` локально і передати у `--model`.)
Відео з відкритих джерел: https://www.youtube.com/watch?v=YCqBYItgi-Q
Файл приклад вихідного формату XML (CVAT for video 1.1)