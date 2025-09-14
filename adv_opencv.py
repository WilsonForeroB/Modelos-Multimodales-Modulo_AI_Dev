import cv2
import numpy as np
import matplotlib.pyplot as plt


# ======================
# Función para mostrar con matplotlib (compatible con VS Code)
# ======================
def mostrar(titulo, imagen, cmap=None):
    if len(imagen.shape) == 3:  # Imagen en color (BGR → RGB)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        plt.imshow(imagen)
    else:
        plt.imshow(imagen, cmap=cmap if cmap else "gray")
    plt.title(titulo)
    plt.axis("off")
    plt.show()


# ======================
# Funciones avanzadas
# ======================

# 1. Detección de bordes avanzada (Canny)
def bordes_canny(img, t1=100, t2=200):
    return cv2.Canny(img, t1, t2)

# 2. Detección de contornos
def contornos(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copia = img.copy()
    cv2.drawContours(copia, contornos, -1, (0, 255, 0), 2)
    return copia

# 3. Transformación de perspectiva
def perspectiva(img):
    alto, ancho = img.shape[:2]
    pts1 = np.float32([[50, 50], [ancho-50, 50], [50, alto-50], [ancho-50, alto-50]])
    pts2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
    matriz = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matriz, (ancho, alto))

# 4. Efecto de caricatura
def caricatura(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.medianBlur(gris, 5)
    bordes = cv2.adaptiveThreshold(gris, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=bordes)

# 5. Efecto de lápiz (dibujo)
def lapiz(img):
    gris, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return gris

# 6. Detección de esquinas (Harris)
def esquinas(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = np.float32(gris)
    dst = cv2.cornerHarris(gris, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img

# 7. Detección de caras (requiere haarcascade)
def detectar_caras(img, xml_path="haarcascade_frontalface_default.xml"):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + xml_path)
    caras = face_cascade.detectMultiScale(gris, 1.1, 4)
    copia = img.copy()
    for (x, y, w, h) in caras:
        cv2.rectangle(copia, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return copia

# 8. Transformación afín (rotación + escalado + desplazamiento)
def afin(img):
    filas, cols, _ = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, filas))

# 9. Filtro de desenfoque gaussiano fuerte
def desenfoque_gaussiano(img, k=25):
    return cv2.GaussianBlur(img, (k, k), 0)

# 10. Filtro estilo "calor" (colormap)
def mapa_calor(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gris, cv2.COLORMAP_JET)


# ======================
# Ejemplo de uso
# ======================
if __name__ == "__main__":
    
    url = "Colombia.webp"
    img = cv2.imread(url)

    mostrar("Original", img)
    mostrar("Bordes Canny", bordes_canny(img))
    mostrar("Contornos", contornos(img))
    mostrar("Perspectiva", perspectiva(img))
    mostrar("Caricatura", caricatura(img))
    mostrar("Dibujo a lápiz", lapiz(img), cmap="gray")
    mostrar("Esquinas Harris", esquinas(img.copy()))
    mostrar("Detección de caras", detectar_caras(img))
    mostrar("Transformación afín", afin(img))
    mostrar("Desenfoque Gaussiano", desenfoque_gaussiano(img))
    mostrar("Mapa de calor", mapa_calor(img))
