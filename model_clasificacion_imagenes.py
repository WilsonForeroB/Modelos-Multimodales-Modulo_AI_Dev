from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
# ======================
# Utilidad para cargar imágenes
# ======================
def cargar_imagen(url=None, archivo=None):
    if url:
        headers = {"User-Agent": "Mozilla/5.0"}  # finge ser un navegador
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # lanza error si el servidor devuelve 403/404
        return Image.open(BytesIO(resp.content)).convert("RGB")
    elif archivo:
        return Image.open(archivo).convert("RGB")
    else:
        raise ValueError("Debes pasar una URL o un archivo")


# ======================
# 1. Clasificación general (objetos / escenas)
# ======================
def clasificar_imagen(imagen, modelo="google/vit-base-patch16-224", top_k=5):
    clasificador = pipeline("image-classification", model=modelo)
    return clasificador(imagen, top_k=top_k)


# ======================
# 2. Clasificación de NSFW / SFW (contenido seguro)
# ======================
def clasificar_seguridad(imagen, modelo="Falconsai/nsfw_image_detection", top_k=2):
    clasificador = pipeline("image-classification", model=modelo)
    return clasificador(imagen, top_k=top_k)


# ======================
# 3. Clasificación de emociones faciales
# ======================
def clasificar_emociones(imagen, modelo="dima806/facial_emotions_image_detection", top_k=3):
    clasificador = pipeline("image-classification", model=modelo)
    return clasificador(imagen, top_k=top_k)


# ======================
# 4. Clasificación de género (hombre / mujer)
# ======================
def clasificar_genero(imagen, modelo="rizvandwiki/gender-classification", top_k=2):
    clasificador = pipeline("image-classification", model=modelo)
    return clasificador(imagen, top_k=top_k)


# ======================
# 5. Describir imagen en lenguaje natural (Image Captioning)
# ======================
def describir_imagen(imagen, modelo="nlpconnect/vit-gpt2-image-captioning"):
    captioner = pipeline("image-to-text", model=modelo)
    return captioner(imagen)[0]["generated_text"]


# ======================
# 6. Detectar objetos (bounding boxes)
# ======================
def detectar_objetos(imagen, modelo="facebook/detr-resnet-50", threshold=0.9):
    detector = pipeline("object-detection", model=modelo)
    return detector(imagen, threshold=threshold)


# ======================
# 7. Clasificación multimodal (preguntas y respuestas sobre imágenes)
# ======================
def responder_pregunta(imagen, pregunta, modelo="Salesforce/blip-vqa-base"):
    vqa = pipeline("visual-question-answering", model=modelo)
    return vqa({"image": imagen, "question": pregunta})[0]["answer"]

# esto iria en helpers
def dibujar_cajas(img_pil, resultados):
    # Convertir de PIL a numpy (BGR para OpenCV)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    for obj in resultados:
        box = obj["box"]
        label = obj["label"]
        score = obj["score"]

        # Dibujar rectángulo
        cv2.rectangle(img,
                      (int(box["xmin"]), int(box["ymin"])),
                      (int(box["xmax"]), int(box["ymax"])),
                      (0, 255, 0), 2)

        # Texto con clase y score
        cv2.putText(img, f"{label} {score:.2f}",
                    (int(box["xmin"]), int(box["ymin"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    return img

# ======================
# Ejemplo de uso
# ======================
if __name__ == "__main__":

    import warnings
    from transformers.utils import logging

    warnings.filterwarnings("ignore", category=UserWarning)
    logging.set_verbosity_error()

    url = "bulldog.jpg"
    img = cargar_imagen(archivo=url)
    img.show()  # solo para verificar que se abre

    # print("🔹 Clasificación general:")
    # print(clasificar_imagen(img))

    # print("\n🔹 Clasificación de seguridad:")
    # print(clasificar_seguridad(img))

    # print("\n🔹 Clasificación de emociones:")
    # print(clasificar_emociones(img))

    # print("\n🔹 Clasificación de género:")
    # print(clasificar_genero(img))

    # print("\n🔹 Descripción de la imagen:")
    # print(describir_imagen(img))

    print("\n🔹 Detección de objetos:")
    resultados = detectar_objetos(img)   # 👈 guardar en variable
    print(resultados)

    # Ahora aplicamos la función que dibuja las cajas
    img_con_cajas = dibujar_cajas(img.copy(), resultados)

    # Mostrar la imagen con cajas (usando matplotlib para VS Code)
    import matplotlib.pyplot as plt
    import cv2

    plt.imshow(cv2.cvtColor(img_con_cajas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

