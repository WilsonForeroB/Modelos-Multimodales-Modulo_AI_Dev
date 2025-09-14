from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import requests as req
from io import BytesIO

# Cargar imagen de ejemplo
def cargar_imagen(url):
    respuesta = req.get(url).content
    return Image.open(BytesIO(respuesta))


# 1. Rotar
def girar(imagen, grados=90):
    return imagen.rotate(grados)


# 2. Voltear horizontalmente
def voltear_horizontal(imagen):
    return imagen.transpose(Image.FLIP_LEFT_RIGHT)


# 3. Voltear verticalmente
def voltear_vertical(imagen):
    return imagen.transpose(Image.FLIP_TOP_BOTTOM)


# 4. Escalar / Redimensionar
def redimensionar(imagen, ancho, alto):
    return imagen.resize((ancho, alto))


# 5. Recortar
def recortar(imagen, caja=(50, 50, 200, 200)):
    return imagen.crop(caja)  
    # caja = (x_inicial, y_inicial, x_final, y_final)


# 6. Convertir a escala de grises
def escala_grises(imagen):
    return imagen.convert("L")


# 7. Invertir colores
def invertir_colores(imagen):
    return ImageOps.invert(imagen.convert("RGB"))


# 8. Mejorar brillo
def ajustar_brillo(imagen, factor=1.5):
    enhancer = ImageEnhance.Brightness(imagen)
    return enhancer.enhance(factor)


# 9. Mejorar contraste
def ajustar_contraste(imagen, factor=1.5):
    enhancer = ImageEnhance.Contrast(imagen)
    return enhancer.enhance(factor)


# 10. Ajustar saturación (color)
def ajustar_color(imagen, factor=1.5):
    enhancer = ImageEnhance.Color(imagen)
    return enhancer.enhance(factor)


# 11. Aplicar desenfoque
def desenfocar(imagen):
    return imagen.filter(ImageFilter.BLUR)


# 12. Detección de bordes
def bordes(imagen):
    return imagen.filter(ImageFilter.FIND_EDGES)


# 13. Convertir a blanco y negro (binaria)
def blanco_y_negro(imagen, umbral=128):
    return imagen.convert("L").point(lambda x: 255 if x > umbral else 0, "1")


# 14. Añadir marco
def marco(imagen, grosor=20, color="red"):
    return ImageOps.expand(imagen, border=grosor, fill=color)


# 15. Guardar imagen
def guardar(imagen, nombre="resultado.png"):
    imagen.save(nombre)
    print(f"Imagen guardada como {nombre}")


# ===============================
# Ejemplo de uso
# ===============================
if __name__ == "__main__":

    url = "Colombia.webp"
    
    img = Image.open(fp=url)
    #img = cargar_imagen(url)

    img_girada = girar(img, 45)
    img_girada.show()

    img_grises = escala_grises(img)
    img_grises.show()


