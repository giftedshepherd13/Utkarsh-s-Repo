# Simple completion
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


prompt = "Complete this sentence: 'I want to'"
response = generator(prompt, max_length=50)
print(response[0]["generated_text"])

# Summarization example
text = """NASA's Perseverance rover successfully landed on Mars as part of the Mars Exploration Program.
It is designed to search for signs of ancient life, collect rock samples, and prepare for future missions."""
summary = generator(f"Summarize: {text}", max_length=50, min_length=20, do_sample=False)
print(summary[0]["generated_text"])

# Question and answer example
question = (
   "The capital of France is Paris. "
   "The Eiffel Tower is located in Paris.\n\n"
   "Question: Where is the Eiffel Tower located?"
)


response = generator(question, max_length=50)
print(response[0]["generated_text"])