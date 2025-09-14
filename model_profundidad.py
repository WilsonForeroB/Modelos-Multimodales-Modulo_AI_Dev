from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

# ======================
# Cargar imagen
# ======================
def cargar_imagen(url=None, archivo=None):
    if url:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    elif archivo:
        return Image.open(archivo).convert("RGB")
    else:
        raise ValueError("Debes pasar una URL o un archivo")


# ======================
# Estimar profundidad
# ======================
def estimar_profundidad(imagen, modelo="Intel/dpt-hybrid-midas"):
    estimator = pipeline("depth-estimation", model=modelo)
    salida = estimator(imagen)
    depth_pil = salida["depth"]
    depth_np = np.array(depth_pil)
    return depth_np


# ======================
# Funciones avanzadas
# ======================

# 1. Normalizar el mapa de profundidad (0-255)
def normalizar_mapa(depth_map):
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_norm.astype("uint8")

# 2. Aplicar colormap (para visualizar mejor)
def aplicar_colormap(depth_map, cmap=cv2.COLORMAP_INFERNO):
    depth_norm = normalizar_mapa(depth_map)
    return cv2.applyColorMap(depth_norm, cmap)

# 3. Superponer el mapa de profundidad sobre la imagen original
def overlay_profundidad(img_pil, depth_map, alpha=0.5):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    depth_color = aplicar_colormap(depth_map)
    overlay = cv2.addWeighted(img, 1 - alpha, depth_color, alpha, 0)
    return overlay

# 4. Segmentar objetos por distancia (ej. primeros planos vs fondo)
def segmentar_por_profundidad(depth_map, threshold=128):
    depth_norm = normalizar_mapa(depth_map)
    _, mask = cv2.threshold(depth_norm, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask

# 5. Simular desenfoque de fondo (bokeh)
def desenfoque_fondo(img_pil, depth_map, threshold=128):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    mask = segmentar_por_profundidad(depth_map, threshold)
    fondo = cv2.GaussianBlur(img, (25, 25), 0)
    resultado = np.where(mask[..., None] == 255, img, fondo)
    return resultado

# 6. Generar nube de puntos 3D (simplificada)
def nube_puntos(depth_map, step=10):
    puntos = []
    for y in range(0, depth_map.shape[0], step):
        for x in range(0, depth_map.shape[1], step):
            z = depth_map[y, x]
            puntos.append((x, y, z))
    return np.array(puntos)


# ======================
# Ejemplo de uso
# ======================
if __name__ == "__main__":
    url = "bulldog.jpg"  # bulldog
    img = cargar_imagen(archivo=url)

    # Estimar profundidad
    depth_map = estimar_profundidad(img)

    # Visualizaciones avanzadas
    depth_color = aplicar_colormap(depth_map)
    overlay = overlay_profundidad(img, depth_map)
    mask = segmentar_por_profundidad(depth_map)
    bokeh = desenfoque_fondo(img, depth_map)

    # Mostrar resultados
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Imagen original")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(depth_map, cmap="gray")
    axs[0, 1].set_title("Mapa de profundidad (grises)")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title("Mapa con colormap")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Overlay profundidad")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(mask, cmap="gray")
    axs[1, 1].set_title("Segmentación por profundidad")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(cv2.cvtColor(bokeh, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Bokeh artificial (fondo desenfocado)")
    axs[1, 2].axis("off")

    plt.show()

    # Ejemplo de nube de puntos
    puntos = nube_puntos(depth_map)
    print(f"Generados {len(puntos)} puntos 3D (submuestreados)")
