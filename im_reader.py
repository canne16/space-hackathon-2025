import numpy as np
import matplotlib.pyplot as plt

def read_image_from_txt(filename):
    """Чтение изображения из текстового файла"""
    with open(filename, 'r') as file:
        # Читаем все строки и преобразуем в массив numpy
        image_data = np.array([[int(num) for num in line.split()] for line in file])
        print(np.mean(image_data), np.std(image_data))
    return image_data

def display_image(image_data):
    """Отображение изображения с помощью matplotlib"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='gray', vmin=0, vmax=512)  # Устанавливаем диапазон значений
    plt.colorbar()
    plt.axis('off')  # Скрываем оси
    plt.show()

# Основная часть программы
if __name__ == "__main__":
    #filename = "sharp_rounded_clouds.txt"  # Например, "image.txt"
    filename = "problem_data/images/film/1.txt"  # Например, "image.txt"

    
    try:
        # Чтение и отображение изображения
        image = read_image_from_txt(filename)
        display_image(image)
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")