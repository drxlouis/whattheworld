import os
import time
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import requests
import io
from PIL import Image

# Load environment variables from a .env file
load_dotenv()

# Define your OpenAI API key
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not OPEN_AI_KEY:
    raise ValueError("OpenAI API key not found. Please set it in your environment variables or .env file.")

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    temperature=0.5,
    openai_api_key=OPEN_AI_KEY,
    model_name="gpt-4",  # Make sure the model name is correct
    max_tokens=100
)

# Function to generate a conversation
def generate_conversation(detections):
    detected_objects = ", ".join(detections)
    messages = [
        SystemMessage(content="Your an intelligent agent that descibes what happends in a museum. You make sure you include every object/person in the scene. You need to give a clear describtion because your describtion is used in a generative ai. Max 70 words"),
        HumanMessage(content=f"List of objects/people in the scene: human, phone")
    ]

    result = llm(messages)
    return result.content

# Function to query the Hugging Face API
def query_huggingface_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/minimaxir/sdxl-wrong-lora"
    headers = {"Authorization": "Bearer " + HUGGINGFACE_API_KEY}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    response.raise_for_status()  # Raise an error for bad responses
    return response.content

# Main loop
while True:
    if os.path.exists("detected_objects.txt"):
        with open("detected_objects.txt", "r") as file:
            detected_objects = file.read().strip().split(": ")[1].split(", ")
            description = generate_conversation(detected_objects)
            print(f"Generated conversation: {description}")
            
            # Export the generated conversation to a file
            with open("detected_objects_conversation.txt", "a") as conv_file:
                conv_file.write(f"{description}\n")
            
            # Generate image based on description
            image_bytes = query_huggingface_api(description)
            
            # Load and display the image
            image = Image.open(io.BytesIO(image_bytes))
            image.show()
            
            # Optionally save the image
            image.save("generated_image.png")
    time.sleep(30)
