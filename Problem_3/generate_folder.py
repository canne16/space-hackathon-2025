import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, laplace
from noise import pnoise2
from skimage.filters import gaussian
from PIL import Image

# Параметры изображения
SIZE = 2000
SCALE = 100
MEAN_BACKGROUND = 100
MEAN_CLOUDS = 511
ALPHA = 30  # Угол движения облаков (в градусах от вертикали)
STEP_SIZE = 5  # Шаг смещения облаков в пикселях
NUM_IMAGES = 10  # Количество изображений

# Параметры генерации облаков
CLOUD_SHARPNESS = 0.65
OCTAVES = 5
PERSISTENCE = 0.05
LACUNARITY = 2.1
EDGE_HARDNESS = 8
BASE_FREQ = 0.8

# Параметры дефектных пикселей
DEFECT_RATIO = 0.001  # 0.1% площади
DEFECT_MIN = 480

NOISE_DISP = 10 / 2 ** 0.5

# Параметры движущихся кружков
CIRCLE_DIAMETER = 10  # Диаметр кружка в пикселях
NUM_CIRCLES = 42  # Количество кружков
CIRCLE_MAX_VALUE = 511  # Максимальная интенсивность
CIRCLE_ANGLE = 45  # Угол движения кружков (градусы)
CIRCLE_SPEED = 3  # Скорость движения кружков (пикселей/кадр)
CIRCLE_ANGLES = [60, 60, 60]  # Углы движения для трех типов
CIRCLE_SPEEDS = [3, 3, 3]  # Скорости для трех типов
CIRCLE_INTENSITIES = [100, 25, 4]  # Интенсивности для трех типов
CIRCLE_INTENSITIES = list(np.array(CIRCLE_INTENSITIES) * NOISE_DISP * 2 ** 0.5)



LAMBDA = 1.5 # Микрометров
PIX_DIAM = round(1.22 * LAMBDA / 4000 * 32000 / 5)
CIRCLE_DIAMETER = PIX_DIAM * 4

def generate_noisy_background(size, mean=100, seed=None):
    """Генерация фона с шумами (с возможностью указания seed)"""
    if seed is not None:
        np.random.seed(seed)
    
    noise = (
        np.random.normal(0, NOISE_DISP, (size, size)) +
        cauchy.rvs(0, 0, size=(size, size)) +  # scale=0 делает распределение дельта-функцией
        laplace.rvs(0, NOISE_DISP, size=(size, size))
    )
    return np.clip(noise + mean, 0, 512)


def generate_perlin_clouds(size, scale, mean, offset_x=0, offset_y=0):
    """Генерация облаков со смещением"""
    clouds = np.zeros((size, size))
    
    x, y = np.meshgrid((np.arange(size) + offset_x)/scale*BASE_FREQ, 
                       (np.arange(size) + offset_y)/scale*BASE_FREQ)
    
    base_noise = np.vectorize(pnoise2)(x, y, 
                                     octaves=OCTAVES,
                                     persistence=PERSISTENCE,
                                     lacunarity=LACUNARITY,
                                     repeatx=size,
                                     repeaty=size)
    
    base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())
    mask = base_noise > CLOUD_SHARPNESS
    
    blurred = gaussian(mask.astype(float), sigma=1)
    edges = np.abs(blurred - gaussian(blurred, sigma=EDGE_HARDNESS/10))
    edge_mask = (edges > 0.2).astype(float)
    
    final_mask = np.clip(mask.astype(float) + edge_mask*0.7, 0, 1)
    
    return final_mask * (mean - MEAN_BACKGROUND) + MEAN_BACKGROUND

def create_defect_mask(size, defect_ratio):
    """Создает маску дефектных пикселей"""
    total_pixels = size * size
    num_defects = int(total_pixels * defect_ratio)
    print(num_defects)
    defect_mask = np.zeros(total_pixels, dtype=bool)
    defect_positions = np.random.choice(total_pixels, num_defects, replace=False)
    defect_mask[defect_positions] = True
    return defect_mask.reshape(size, size)

def create_circle(radius, int_num):
    """Создает кружок с заданным распределением интенсивности"""
    diameter = 2 * radius + 1
    x = np.linspace(-np.pi, np.pi, diameter)
    y = np.linspace(-np.pi, np.pi, diameter)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    
    # Распределение интенсивности по закону (sin(x)/x)^2
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = (np.sin(r)/r)**2
    intensity[r == 0] = 1.0  # Устраняем деление на 0 в центре
    
    return int_num * intensity

def generate_circles(size, num_circles, radius, frame_num, all_circle_positions=None):
    """Генерирует движущиеся кружки Эйри трех типов и возвращает их позиции"""
    circles_layer = np.zeros((size, size))
    radius = CIRCLE_DIAMETER // 2
    circles_per_type = num_circles // 3
    
    # Создаем три типа кругов Эйри
    airy_disks = [
        create_circle(radius, CIRCLE_INTENSITIES[0]),
        create_circle(radius, CIRCLE_INTENSITIES[1]),
        create_circle(radius, CIRCLE_INTENSITIES[2])
    ]
    
    # Начальные позиции кружков (случайные)
    if frame_num == 0:
        initial_positions = np.random.randint(0, size, (num_circles, 2))
        np.save("circle_positions.npy", initial_positions)
    else:
        initial_positions = np.load("circle_positions.npy")
    
    # Массив для хранения текущих позиций кружков
    current_positions = []
    
    for i in range(num_circles):
        # Определяем тип круга (0, 1 или 2)
        circle_type = i // circles_per_type
        if circle_type > 2: circle_type = 2
        
        # Параметры движения для данного типа
        angle_rad = np.deg2rad(CIRCLE_ANGLES[circle_type])
        dx = int(CIRCLE_SPEEDS[circle_type] * np.cos(angle_rad))
        dy = int(CIRCLE_SPEEDS[circle_type] * np.sin(angle_rad))
        
        # Текущая позиция
        pos = initial_positions[i]
        x_pos = (pos[0] + frame_num * dx) % size
        y_pos = (pos[1] + frame_num * dy) % size
        
        # Сохраняем текущую позицию
        current_positions.append((x_pos, y_pos, circle_type))
        
        # Размещаем кружок на изображении
        circle = airy_disks[circle_type]
        c_size = circle.shape[0]
        
        for i in range(c_size):
            for j in range(c_size):
                x = (x_pos + i - radius) % size
                y = (y_pos + j - radius) % size
                circles_layer[x, y] = max(circles_layer[x, y], circle[i, j])
    
    # Если указан словарь для сохранения позиций, добавляем текущие позиции
    if all_circle_positions is not None:
        all_circle_positions[frame_num] = current_positions
    
    return circles_layer

def generate_image_sequence():
    """Генерация последовательности изображений с сохранением координат кружков"""
    # Создаем папки для результатов
    os.makedirs("noise", exist_ok=True)
    os.makedirs("film", exist_ok=True)
    
    # 1. Сохраняем 2 изображения с только фоновым шумом в папку noise
    for i in range(2):
        noise_img = generate_noisy_background(SIZE, MEAN_BACKGROUND, seed=i)
        np.savetxt(f"noise/noise_{i:02d}.txt", noise_img, fmt='%d')
        plt.imsave(f"noise/noise_{i:02d}.png", noise_img, cmap='gray', vmin=0, vmax=512)
        print(f"Создан шумовой образец {i+1}/2 в папке noise")
    
    # 2. Генерируем последовательность для папки film
    defect_mask = create_defect_mask(SIZE, DEFECT_RATIO)
    defect_values = np.full(defect_mask.sum(), DEFECT_MIN, dtype=np.int32)
    
    rad = np.deg2rad(ALPHA)
    steps_x = [int(np.sin(rad) * STEP_SIZE * i) for i in range(NUM_IMAGES)]
    steps_y = [int(np.cos(rad) * STEP_SIZE * i) for i in range(NUM_IMAGES)]
    
    # Для создания GIF
    gif_images = []
    
    # Словарь для хранения координат кружков на каждом кадре
    all_circle_positions = {}
    
    for i in range(NUM_IMAGES):
        background = generate_noisy_background(SIZE, MEAN_BACKGROUND, seed=i)
        clouds = generate_perlin_clouds(SIZE, SCALE, MEAN_CLOUDS, steps_x[i], steps_y[i])
        circles = generate_circles(SIZE, NUM_CIRCLES, CIRCLE_DIAMETER//2, i, all_circle_positions)
        
        image = np.where(clouds > MEAN_BACKGROUND + 30, clouds, background)
        image = np.where(circles > 0, np.maximum(image, circles), image)
        image[defect_mask] = defect_values
        image = np.clip(image, 0, 512).astype(np.int32)
        
        # Сохранение в папку film
        np.savetxt(f"film/frame_{i:02d}.txt", image, fmt='%d')
        plt.imsave(f"film/frame_{i:02d}.png", image, cmap='gray', vmin=0, vmax=512)
        
        # Добавляем кадр для GIF
        gif_images.append(Image.fromarray((image * 255 / 512).astype(np.uint8)))
        
        if i == 0:
            print(f"Среднее значение фона: {np.mean(image[~defect_mask & (image <= MEAN_BACKGROUND + 30)]):.1f}")
            print(f"Среднее значение облаков: {np.mean(image[~defect_mask & (image > MEAN_BACKGROUND + 30)]):.1f}")
        
        print(f"Создан кадр {i+1}/{NUM_IMAGES} в папке film")
    
    # Сохраняем GIF
    gif_images[0].save('film/animation.gif',
                      save_all=True,
                      append_images=gif_images[1:],
                      duration=200,  # мс между кадрами
                      loop=0)  # бесконечный цикл
    
    # Сохраняем координаты кружков
    # Создаем список треков в формате, удобном для сравнения с предсказанными траекториями
    tracks_list = []
    # Группируем координаты кружков по типам
    for circle_type in range(3):
        frame_tracks = []
        for frame_num in range(NUM_IMAGES):
            # Фильтруем позиции кружков по типу
            frame_positions = [(x, y) for x, y, t in all_circle_positions[frame_num] if t == circle_type]
            frame_tracks.append(frame_positions)
        tracks_list.append(frame_tracks)
    
    # Сохраняем все позиции в удобном формате
    np.save("film/true_tracks.npy", tracks_list)
    
    # Также сохраняем все позиции в оригинальном виде для дополнительной информации
    np.save("film/all_circle_positions.npy", all_circle_positions)
    
    # Удаляем временный файл с начальными позициями
    if os.path.exists("circle_positions.npy"):
        os.remove("circle_positions.npy")
    
    # Создадим визуализацию траекторий для справки
    visualize_true_trajectories(all_circle_positions, NUM_IMAGES)
    
    print("Созданы все файлы:")
    print(f"- 2 шумовых образца в папке noise")
    print(f"- {NUM_IMAGES} кадров в папке film")
    print(f"- Анимация animation.gif в папке film")
    print(f"- Координаты кружков сохранены в film/true_tracks.npy")
    print(f"- Подробные данные о кружках сохранены в film/all_circle_positions.npy")
    print(f"- Визуализация истинных траекторий сохранена в film/true_trajectories.png")

def visualize_true_trajectories(all_positions, num_frames):
    """Визуализация истинных траекторий кружков"""
    plt.figure(figsize=(12, 12))
    
    # Создаем цвета для трех типов кружков
    colors = ['red', 'green', 'blue']
    
    # Определяем, сколько кружков каждого типа
    circle_count = len(all_positions[0])
    circles_per_type = circle_count // 3
    
    # Создаем словарь трекеров (circle_id: [позиции])
    tracks = {}
    
    for frame in range(num_frames):
        for i, (x, y, circle_type) in enumerate(all_positions[frame]):
            if i not in tracks:
                tracks[i] = []
            tracks[i].append((x, y, circle_type))
    
    # Отрисовываем треки
    for circle_id, positions in tracks.items():
        # Получаем тип кружка
        circle_type = positions[0][2]
        color = colors[circle_type]
        
        # Разделяем координаты
        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        
        # Рисуем линию трека
        plt.plot(xs, ys, '-', color=color, linewidth=1, alpha=0.7, 
                label=f'Тип {circle_type+1}' if circle_id % circles_per_type == 0 else None)
        
        # Отмечаем начальную и конечную точки
        plt.plot(xs[0], ys[0], 'o', color=color, markersize=6)
        plt.plot(xs[-1], ys[-1], 's', color=color, markersize=6)
    
    plt.title('Истинные траектории движения кружков')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Установка границ изображения
    plt.xlim(0, SIZE)
    plt.ylim(0, SIZE)
    
    # Сохраняем визуализацию
    plt.savefig('film/true_trajectories.png', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_image_sequence()
    print("Все изображения успешно сохранены в папку cloud_sequence")
