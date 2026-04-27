# Makkale, ithu puthu library! Install panna maranthudathinga:
# pip install transformers
from transformers import pipeline

print("Loading the AI Brain... (First time run aaga konjam time aagum ⏳)")

# 1. Download & Load the AI Model
# Inga namma OpenAI-oda 'gpt2' model-ah use panrom. 
# 'text-generation' na puthusa ezhuthi thara solrom nu artham.
ai_writer = pipeline('text-generation', model='gpt2')

# 2. Namma Input Prompt (AI kitta enna ezhutha solla porom?)
my_prompt = "Artificial Intelligence will change the world by"

print(f"\n🗣️ You: {my_prompt}...\n")
print("🤖 AI is thinking and writing...\n")

# 3. Generating the Text
# max_length=50 na 50 varthaigal ezhutha solrom
# num_return_sequences=1 na 1 paragraph mattum pothum
result = ai_writer(my_prompt, max_length=50, num_return_sequences=1, truncation=True)

# 4. Final Output
print("--- AI Generated Output ---")
print(result[0]['generated_text'])
print("---------------------------")