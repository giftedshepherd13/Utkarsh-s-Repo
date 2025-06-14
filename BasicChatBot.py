# Simple completion
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Basic chatbot
chatbot_prompt = "You are a friendly AI assistant. Answer the user’s question with a helpful response."
messages = [{"role": "user", "content": "Tell me a fact about the Sun."}]
response = generator(f"{chatbot_prompt} {messages[-1]['content']}", max_length=50)
print(response[0]["generated_text"])