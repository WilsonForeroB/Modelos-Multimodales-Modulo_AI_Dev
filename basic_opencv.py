import cv2
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# ======================
# Cargar imagen desde URL
# ======================
def cargar_imagen(url):
    resp = requests.get(url)
    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # formato BGR


# ======================
# Funciones básicas
# ======================

# 1. Mostrar imagen
def mostrar(titulo, imagen):
    if len(imagen.shape) == 3:  # Color → convertir BGR a RGB
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        plt.imshow(imagen)
    else:  # Escala de grises
        plt.imshow(imagen, cmap="gray")
    plt.title(titulo)
    plt.axis("off")
    plt.show()

# 2. Escala de grises
def escala_grises(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Redimensionar
def redimensionar(img, ancho=200, alto=200):
    return cv2.resize(img, (ancho, alto))

# 4. Rotar
def rotar(img, grados=90):
    alto, ancho = img.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz = cv2.getRotationMatrix2D(centro, grados, 1.0)
    return cv2.warpAffine(img, matriz, (ancho, alto))

# 5. Recortar
def recortar(img, x1=50, y1=50, x2=200, y2=200):
    return img[y1:y2, x1:x2]

# 6. Invertir colores
def invertir_colores(img):
    return cv2.bitwise_not(img)

# 7. Desenfoque (Blur)
def desenfocar(img, k=15):
    return cv2.GaussianBlur(img, (k, k), 0)

# 8. Detección de bordes
def bordes(img):
    gris = escala_grises(img)
    return cv2.Canny(gris, 100, 200)

# 9. Guardar imagen
def guardar(img, nombre="resultado.jpg"):
    cv2.imwrite(nombre, img)
    print(f"Imagen guardada como {nombre}")


# ======================
# Ejemplo de uso
# ======================
if __name__ == "__main__":

    url = "Colombia.webp"
    #img = cargar_imagen()
    img = cv2.imread(filename=url)

    # Ejemplos
    gris = escala_grises(img)
    pequeño = redimensionar(img, 150, 150)
    girada = rotar(img, 45)
    recortada = recortar(img)
    invertida = invertir_colores(img)
    blur = desenfocar(img)
    edge = bordes(img)

    # Mostrar resultados
    mostrar("Original", img)
    mostrar("Grises", gris)
    mostrar("Redimensionada", pequeño)
    mostrar("Rotada", girada)
    mostrar("Recortada", recortada)
    mostrar("Invertida", invertida)
    mostrar("Desenfoque", blur)
    mostrar("Bordes", edge)

    # Guardar ejemplo
    guardar(invertida, "Test/invertida.jpg")
