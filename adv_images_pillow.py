from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont
import requests as req
from io import BytesIO
import math


# Cargar imagen desde URL
def cargar_imagen(url):
    respuesta = req.get(url).content
    return Image.open(BytesIO(respuesta)).convert("RGBA")


# 1. Rotación con fondo transparente
def rotacion_transparente(imagen, grados=45):
    return imagen.rotate(grados, expand=True, fillcolor=(0, 0, 0, 0))


# 2. Crear sombra alrededor
def sombra(imagen, offset=(10, 10), background="white", shadow_color="black"):
    total_width = imagen.width + abs(offset[0]) + 20
    total_height = imagen.height + abs(offset[1]) + 20

    fondo = Image.new("RGBA", (total_width, total_height), background)
    sombra = Image.new("RGBA", imagen.size, shadow_color)
    sombra = sombra.filter(ImageFilter.GaussianBlur(10))

    fondo.paste(sombra, (offset[0] + 10, offset[1] + 10), sombra)
    fondo.paste(imagen, (10, 10), imagen)
    return fondo


# 3. Redondear esquinas
def esquinas_redondeadas(imagen, radio=50):
    mask = Image.new("L", imagen.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), imagen.size], radius=radio, fill=255)
    imagen.putalpha(mask)
    return imagen


# 4. Crear un círculo con la imagen
def recorte_circular(imagen):
    mask = Image.new("L", imagen.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([(0, 0), imagen.size], fill=255)
    imagen.putalpha(mask)
    return imagen


# 5. Añadir texto con fuente personalizada
def agregar_texto(imagen, texto="Hola HG.F.", posicion=(50, 50), tamaño=40, color="red"):
    draw = ImageDraw.Draw(imagen)
    try:
        font = ImageFont.truetype("arial.ttf", tamaño)  # Necesitas la fuente instalada
    except:
        font = ImageFont.load_default()
    draw.text(posicion, texto, font=font, fill=color)
    return imagen


# 6. Efecto espejo (reflejo en agua)
def reflejo(imagen, factor=0.5):
    reflejo = imagen.transpose(Image.FLIP_TOP_BOTTOM)
    reflejo = reflejo.crop((0, 0, imagen.width, int(imagen.height * factor)))
    gradiente = Image.new("L", (1, reflejo.height), color=255)
    for y in range(reflejo.height):
        gradiente.putpixel((0, y), max(0, 255 - int(255 * (y / reflejo.height))))
    alpha = gradiente.resize(reflejo.size)
    reflejo.putalpha(alpha)

    resultado = Image.new("RGBA", (imagen.width, imagen.height + reflejo.height))
    resultado.paste(imagen, (0, 0))
    resultado.paste(reflejo, (0, imagen.height), reflejo)
    return resultado


# 7. Mosaico (efecto pixelado)
def pixelar(imagen, factor=10):
    pequeña = imagen.resize((imagen.width // factor, imagen.height // factor), resample=Image.NEAREST)
    return pequeña.resize(imagen.size, Image.NEAREST)


# 8. Distorsión tipo onda
def onda(imagen, amplitud=20, frecuencia=2):
    ancho, alto = imagen.size
    resultado = Image.new("RGBA", (ancho, alto))
    pixeles = imagen.load()
    resultado_pix = resultado.load()

    for y in range(alto):
        for x in range(ancho):
            nuevo_x = int(x + amplitud * math.sin(2 * math.pi * y / frecuencia))
            if 0 <= nuevo_x < ancho:
                resultado_pix[x, y] = pixeles[nuevo_x, y]
    return resultado


# 9. Superponer dos imágenes
def superponer(imagen1, imagen2, alpha=0.5):
    imagen2 = imagen2.resize(imagen1.size)
    return Image.blend(imagen1.convert("RGBA"), imagen2.convert("RGBA"), alpha)


# 10. Guardar como WebP con compresión
def guardar_webp(imagen, nombre="resultado.webp"):
    imagen.save(nombre, "WEBP", quality=80)
    print(f"Imagen guardada como {nombre}")

# ===============================
# Ejemplo de uso
# ===============================
if __name__ == "__main__":
    url = "Colombia.webp"
    
    img = Image.open(fp=url)
    #img = cargar_imagen(url)

    imagen_onda = onda(img)
    imagen_onda.show()

    imagen_reflejo = reflejo(img)
    imagen_reflejo.show()

    #guardar
    guardar_webp(imagen_reflejo, "Test/reflejo.png")