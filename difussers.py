import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ======================
# CONFIGURACIÓN GENERAL
# ======================
# Modelo base (puedes cambiar a otro como SDXL, OpenJourney, etc.)
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Prompt principal
PROMPT = "A photorealistic English bulldog wearing a wedding tuxedo, elegant, highly detailed, cinematic lighting, 4k"

# Dispositivo: usa "cuda" si tienes GPU, "cpu" si no
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inicializar pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)


# ======================
# FUNCIONES DE VARIACIONES
# ======================

# 1. Generar imagen básica
def generar_basico(prompt=PROMPT):
    return pipe(prompt).images[0]

# 2. Cambiar número de pasos (más pasos = más detalle)
def generar_con_pasos(prompt=PROMPT, steps=50):
    return pipe(prompt, num_inference_steps=steps).images[0]

# 3. Cambiar guidance scale (controla fidelidad al prompt)
def generar_con_guidance(prompt=PROMPT, guidance=8.5):
    return pipe(prompt, guidance_scale=guidance).images[0]

# 4. Generar múltiples imágenes
def generar_multiples(prompt=PROMPT, n=4, steps=40):
    return pipe([prompt] * n, num_inference_steps=steps).images

# 5. Usar un modelo distinto (ej: SDXL)
def generar_con_modelo(prompt=PROMPT, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    pipe_alt = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    return pipe_alt(prompt, num_inference_steps=50, guidance_scale=8).images[0]

# ======================
# FLUJO CONTINUO
# ======================
if __name__ == "__main__":

    # 👉 Por defecto: generar imagen básica
    image = generar_basico()

    # 2. Con más pasos (mejor calidad, más lento)
    # image = generar_con_pasos(steps=75)

    # 3. Con guidance scale (control de fidelidad al prompt)
    # image = generar_con_guidance(guidance=10)

    # 4. Varias imágenes a la vez
    # images = generar_multiples(n=4)
    # for i, img in enumerate(images):
    #     img.save(f"bulldog_variacion_{i}.png")

    # 5. Cambiar modelo (ej. SDXL)
    # image = generar_con_modelo()

    # ======================
    # Guardar resultado
    # ======================
    image.save("Test/bulldog.png")
    image.show()
