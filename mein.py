import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
from PIL import Image
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ====== CARGAR VARIABLES DE ENTORNO ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "org-Gpagyx25nqJdVGVZUTiZBGUY")

if not OPENAI_API_KEY:
    raise ValueError("No se encontró la API Key en el archivo .env")

# ====== INICIALIZAR CLIENTE OPENAI ======
client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID)

# ====== CARGAR MODELO RESNET50 ======
model = resnet50(pretrained=True)
model.eval()

# ====== TRANSFORMACIÓN DE IMAGEN ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ====== CLASES DEL DATASET FOOD-101 ======
response = requests.get(
    "https://raw.githubusercontent.com/taspinar/tensorflow_food_classification/master/food-101/classes.txt"
)
if response.status_code == 200:
    food_classes = [line.strip() for line in response.text.splitlines()]
else:
    food_classes = ["unknown"] * 101

# ====== FUNCIÓN: CLASIFICAR COMIDA ======
def classify_food(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    idx = predicted.item()
    print(f"[DEBUG] Predicción índice: {idx} | Total clases: {len(food_classes)}")

    return food_classes[idx] if 0 <= idx < len(food_classes) else "Clase desconocida"

# ====== FUNCIÓN: INFORMACIÓN NUTRICIONAL SIMULADA ======
def simulate_nutrition(food_name):
    base = abs(hash(food_name)) % 300 + 100
    return {
        "Calorías": f"{base} kcal",
        "Proteínas": f"{base // 10} g",
        "Grasas": f"{base // 12} g",
        "Carbohidratos": f"{base // 8} g"
    }

# ====== FUNCIÓN: GENERAR RECETA ======
def generate_recipe_with_ai(food_name):
    prompt = f"Dame una receta sencilla y nutritiva usando {food_name} como ingrediente principal."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un chef experto en recetas nutritivas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al llamar a la API de OpenAI:\n{str(e)}"

# ====== FUNCIÓN: CHAT LIBRE ======
def chat_with_ai(user_message):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en nutrición y comida."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error en respuesta IA:\n{str(e)}"

# ====== FUNCIÓN PRINCIPAL ======
def food_app(image, user_message=""):
    label = classify_food(image)
    nutrition = simulate_nutrition(label)
    recipe = generate_recipe_with_ai(label)

    extra_info = chat_with_ai(user_message) if user_message.strip() else ""

    result = f"##  Comida Detectada: **{label}**\n\n"
    result += "###  Información Nutricional Estimada:\n"
    for k, v in nutrition.items():
        result += f"- {k}: {v}\n"
    result += f"\n###  Receta Sugerida:\n{recipe}\n"
    if extra_info:
        result += f"\n###  Respuesta de la IA:\n{extra_info}"
    return result

# ====== INTERFAZ GRADIO ======
interface = gr.Interface(
    fn=food_app,
    inputs=[
        gr.Image(type="pil", label=" Sube una imagen de comida"),
        gr.Textbox(label="Pregunta o mensaje para la IA (opcional)", placeholder="Ej: ¿Esta comida es saludable?")
    ],
    outputs="markdown",
    title="🍽 NutriFood AI",
    description="Sube una imagen y recibe una receta, información nutricional y respuesta de la IA."
)
if __name__ == "__main__":
    interface.launch()
