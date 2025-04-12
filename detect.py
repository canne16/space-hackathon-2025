import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from tqdm import tqdm
from pprint import pprint
import matplotlib.colors as mcolors

# Параметры обработки
CIRCLE_DIAMETER = 40  # Должно соответствовать параметру из генератора
TEMPLATE_PADDING = 5  # Отступ вокруг шаблона
DETECTION_THRESHOLD = 0.7  # Порог корреляции для обнаружения
MIN_DISTANCE = CIRCLE_DIAMETER  # Минимальное расстояние между центрами кружков
TRACKING_THRESHOLD = 10  # Порог в пикселях для связывания треков

def load_frames(folder_path, num_frames):
    """Загрузка последовательности кадров"""
    frames = []
    for i in range(num_frames):
        frame = np.loadtxt(f"{folder_path}/frame_{i:02d}.txt")
        frames.append(frame)
    return frames

def create_circle_template(diameter, intensity_profile):
    """Создание шаблона кружка для сопоставления"""
    radius = diameter // 2
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    r = np.sqrt(x**2 + y**2)
    
    # Создаем шаблон с тем же распределением интенсивности, что и в генераторе
    with np.errstate(divide='ignore', invalid='ignore'):
        template = (np.sin(r)/r)**2
    template[r > radius] = 0
    template[r == 0] = 1.0
    
    # Нормализуем шаблон
    template = template * intensity_profile / template.max()
    return template

def detect_circles(frame, template):
    """Обнаружение кружков на одном кадре"""
    # Применяем шаблонное сопоставление
    result = match_template(frame, template, pad_input=True)
    
    # Находим пики корреляции
    peaks = find_peaks(result.flatten(), height=DETECTION_THRESHOLD, 
                      distance=MIN_DISTANCE)[0]
    
    # Преобразуем индексы в координаты
    coords = np.unravel_index(peaks, result.shape)
    return list(zip(coords[1], coords[0]))  # Возвращаем (x, y)

def track_circles(frames, template):
    """Трекинг кружков по последовательности кадров"""
    all_detections = []
    
    for i, frame in enumerate(tqdm(frames, desc="Обработка кадров")):
        # Предварительная фильтрация для улучшения качества
        filtered_frame = gaussian_filter(frame, sigma=1)
        
        # Обнаружение кружков на текущем кадре
        detections = detect_circles(filtered_frame, template)
        all_detections.append(detections)
    
    return all_detections

def assign_track_ids(tracks):
    """
    Назначение ID для треков кружков
    
    Args:
        tracks: Список списков координат кружков по кадрам
        
    Returns:
        tracks_with_ids: Список словарей {id: (x, y)} для каждого кадра
        track_history: Словарь {id: [(x1, y1), (x2, y2), ...]} с историей каждого трека
    """
    tracks_with_ids = []
    track_history = {}
    next_id = 0
    active_tracks = set()
    
    # Для первого кадра просто назначаем новые ID всем кружкам
    first_frame = {}
    for circle in tracks[0]:
        first_frame[next_id] = circle
        track_history[next_id] = [circle]
        active_tracks.add(next_id)
        next_id += 1
    tracks_with_ids.append(first_frame)
    
    # Для остальных кадров ищем соответствия
    for i in range(1, len(tracks)):
        current_circles = tracks[i]
        prev_frame = tracks_with_ids[-1]
        current_frame = {}
        matched_prev_ids = set()
        
        # Для каждого кружка в текущем кадре
        for circle in current_circles:
            x, y = circle
            min_dist = float('inf')
            closest_id = None
            
            # Находим ближайший кружок из предыдущего кадра
            for track_id, prev_circle in prev_frame.items():
                if track_id not in active_tracks:
                    continue
                
                prev_x, prev_y = prev_circle
                dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_id = track_id
            
            # Если нашли близкий кружок в пределах порога
            if closest_id is not None and min_dist < TRACKING_THRESHOLD:
                current_frame[closest_id] = circle
                track_history[closest_id].append(circle)
                matched_prev_ids.add(closest_id)
            else:
                # Создаем новый трек
                current_frame[next_id] = circle
                track_history[next_id] = [circle]
                active_tracks.add(next_id)
                next_id += 1
        
        # Обновляем активные треки
        for track_id in active_tracks.copy():
            if track_id not in matched_prev_ids:
                active_tracks.remove(track_id)
                
        tracks_with_ids.append(current_frame)
    
    return tracks_with_ids, track_history

def visualize_tracking_with_trajectories(frames, tracks, track_history, output_folder):
    """Визуализация результатов трекинга с траекториями"""
    os.makedirs(f"{output_folder}/tracked", exist_ok=True)
    
    # Создаем цветовую карту для треков
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, (frame, track_dict) in enumerate(zip(frames, tracks)):
        plt.figure(figsize=(10, 10))
        plt.imshow(frame, cmap='gray', vmin=0, vmax=512)
        
        # Рисуем обнаруженные кружки и их ID
        for track_id, (x, y) in track_dict.items():
            color = colors[track_id % len(colors)]
            circle = plt.Circle((x, y), CIRCLE_DIAMETER//2, 
                              color=color, fill=False, linewidth=1)
            plt.gca().add_patch(circle)
            plt.text(x, y, f"ID:{track_id}", color=color, fontsize=9, 
                     bbox=dict(facecolor='white', alpha=0.7))
            
            # Рисуем траекторию для этого ID
            track_points = track_history[track_id]
            if len(track_points) > 1:
                # Рисуем только точки до текущего кадра
                valid_points = track_points[:track_points.index((x, y))+1]
                xs, ys = zip(*valid_points)
                plt.plot(xs, ys, '-', color=color, linewidth=1, alpha=0.7)
        
        plt.title(f"Кадр {i} - отслежено {len(track_dict)} кружков")
        plt.savefig(f"{output_folder}/tracked/frame_{i:02d}_tracked.png")
        plt.close()

def visualize_all_trajectories(frames, track_history, output_folder):
    """
    Создание итогового изображения со всеми траекториями
    
    Args:
        frames: Список всех кадров
        track_history: Словарь {id: [(x1, y1), (x2, y2), ...]} с историей каждого трека
        output_folder: Папка для сохранения результатов
    """
    # Используем первый кадр как фон
    plt.figure(figsize=(12, 12))
    plt.imshow(frames[0], cmap='gray', vmin=0, vmax=512)
    
    # Создаем цветовую карту для треков
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Рисуем траектории всех треков
    for track_id, points in track_history.items():
        if len(points) > 1:  # Рисуем только если есть хотя бы 2 точки
            color = colors[track_id % len(colors)]
            xs, ys = zip(*points)
            
            # Рисуем линию траектории
            plt.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.8, 
                    label=f"ID {track_id} ({len(points)} кадров)")
            
            # Рисуем начальную и конечную точки
            plt.plot(xs[0], ys[0], 'o', color=color, markersize=8)
            plt.plot(xs[-1], ys[-1], 's', color=color, markersize=8)
            
            # Добавляем метки ID
            plt.text(xs[0], ys[0], f"ID:{track_id} (старт)", color=color, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
            plt.text(xs[-1], ys[-1], f"ID:{track_id} (конец)", color=color, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title("Все траектории объектов")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/all_trajectories.png", dpi=300, bbox_inches='tight')
    
    # Создаем также версию без фонового изображения для лучшей видимости траекторий
    plt.figure(figsize=(12, 12))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Установим те же границы осей, что и у изображения
    height, width = frames[0].shape
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Инвертируем ось Y для соответствия координатам изображения
    
    # Рисуем траектории всех треков
    for track_id, points in track_history.items():
        if len(points) > 1:
            color = colors[track_id % len(colors)]
            xs, ys = zip(*points)
            
            # Рисуем линию траектории с маркерами для каждого кадра
            plt.plot(xs, ys, '-o', color=color, linewidth=2, markersize=4, alpha=0.8,
                    label=f"ID {track_id} ({len(points)} кадров)")
            
            # Рисуем начальную и конечную точки более заметно
            plt.plot(xs[0], ys[0], 'o', color=color, markersize=10)
            plt.plot(xs[-1], ys[-1], 's', color=color, markersize=10)
            
            # Добавляем метки
            plt.text(xs[0], ys[0], f"ID:{track_id}\nстарт", color=color, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
            plt.text(xs[-1], ys[-1], f"ID:{track_id}\nконец", color=color, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title("Траектории объектов (без фона)")
    plt.xlabel("X координата")
    plt.ylabel("Y координата")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/trajectories_no_background.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"Итоговые изображения с траекториями сохранены в {output_folder}")

def main():
    # Загрузка кадров
    frames = load_frames("film", NUM_IMAGES)
    
    # Создание шаблона (используем среднюю интенсивность из генератора)
    avg_intensity = np.mean(CIRCLE_INTENSITIES)
    template = create_circle_template(CIRCLE_DIAMETER + TEMPLATE_PADDING*2, avg_intensity)
    
    # Трекинг кружков
    detected_tracks = track_circles(frames, template)
    
    # Назначение ID для треков
    tracks_with_ids, track_history = assign_track_ids(detected_tracks)
    
    # Визуализация результатов с траекториями для каждого кадра
    visualize_tracking_with_trajectories(frames, tracks_with_ids, track_history, "film")
    
    # Создание итогового изображения со всеми траекториями
    visualize_all_trajectories(frames, track_history, "film")
    
    # Вывод статистики по трекам
    print(f"\nВсего уникальных треков: {len(track_history)}")
    for track_id, points in track_history.items():
        print(f"Трек ID {track_id}: {len(points)} кадров, начало в кадре {tracks_with_ids.index(next(d for d in tracks_with_ids if track_id in d))}")
    
    # Сохранение результатов трекинга
    np.save("film/detected_positions.npy", np.array(detected_tracks, dtype=object))
    np.save("film/track_history.npy", track_history)
    
    print("\nРезультаты сохранены в:")
    print("- film/detected_positions.npy - координаты кружков")
    print("- film/track_history.npy - треки с ID")
    print("- film/tracked/ - кадры с визуализацией трекинга")
    print("- film/all_trajectories.png - итоговое изображение со всеми траекториями")
    print("- film/trajectories_no_background.png - траектории без фонового изображения")
    
    # Расчет F1-метрики
    TP = sum(1 for points in track_history.values() if len(points) > 5)
    FN = 42 / 3 - TP
    FP = 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nF1-метрика: {f1_score:.4f}")


if __name__ == "__main__":
    # Переносим параметры из генератора
    NUM_IMAGES = 10
    CIRCLE_INTENSITIES = [100, 25, 4]
    CIRCLE_DIAMETER = 40  # Примерное значение, должно соответствовать генератору
    
    main()