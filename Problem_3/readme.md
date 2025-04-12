## Запуск разметки папок

```
python -m venv .venv
.venv/bin/activate
pip install -r req_rasm.txt
python generate_folder.py
```

## Пример вывода


Создан шумовой образец 1/2 в папке noise
Создан шумовой образец 2/2 в папке noise
4000
Среднее значение фона: 99.1
Среднее значение облаков: 495.1
Создан кадр 1/10 в папке film
Создан кадр 2/10 в папке film
Создан кадр 3/10 в папке film
Создан кадр 4/10 в папке film
Создан кадр 5/10 в папке film
Создан кадр 6/10 в папке film
Создан кадр 7/10 в папке film
Создан кадр 8/10 в папке film
Создан кадр 9/10 в папке film
Создан кадр 10/10 в папке film
Созданы все файлы:
- 2 шумовых образца в папке noise
- 10 кадров в папке film
- Анимация animation.gif в папке film
- Координаты кружков сохранены в film/true_tracks.npy
- Подробные данные о кружках сохранены в film/all_circle_positions.npy
- Визуализация истинных траекторий сохранена в film/true_trajectories.png
Все изображения успешно сохранены в папку cloud_sequence


## Запуск детекции

```
python -m venv .venv
.venv/bin/activate
pip install -r req_det.txt
python detect.py
```

## Пример вывода


Результаты сохранены в:
- film/detected_positions.npy - координаты кружков
- film/track_history.npy - треки с ID
- film/tracked/ - кадры с визуализацией трекинга
- film/all_trajectories.png - итоговое изображение со всеми траекториями
- film/trajectories_no_background.png - траектории без фонового изображения
7

F1-метрика: 0.6667
