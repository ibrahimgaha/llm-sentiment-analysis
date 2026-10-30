
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MODEL_NAME = "mistralai/mistral-large-4-1t-a52b"

model = ChatNVIDIA(
    model=MODEL_NAME,
    temperature=0,
    max_completion_tokens=256,
)

try:
    response = model.invoke(
        "You are a sentiment analysis assistant.\n"
        "Classify the following review as Positive, Negative, "
        "or Neutral.\n"
        "Return only the sentiment label.\n\n"
        "Review: I absolutely love this product!"
    )

    print("Model:", MODEL_NAME)
    print("Sentiment:", response.content)

except Exception as error:
    print("Error:", error)
