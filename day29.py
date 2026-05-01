from transformers import pipeline
print("Loading the AI Brain... (First time run aaga konjam time aagum ⏳)")
ai_writer = pipeline('text-generation', model='gpt2')
my_prompt = "Artificial Intelligence will change the world by"
print(f"\n🗣️ You: {my_prompt}...\n")
print("🤖 AI is thinking and writing...\n")
result = ai_writer(my_prompt, max_length=50, num_return_sequences=1, truncation=True)
print("--- AI Generated Output ---")
print(result[0]['generated_text'])
print("---------------------------")

# Importing the 'pipeline' tool from Hugging Face Transformers, the gold standard for modern AI
# Printing a warning (First time run aaga konjam time aagum!) because it has to download the heavy model files
# Initializing the AI brain! We are using GPT-2 specifically for the 'text-generation' task
# Setting up the starting sentence (the prompt) that we want the AI to finish for us
# Printing out our starting prompt to the terminal so we can see the before-and-after
# Letting the user know the AI is processing...
# The Magic Step: The AI reads our prompt and writes the rest of the sentence!
# 'max_length=50' tells it to stop after 50 words, so it doesn't write an entire book
# --- Displaying the Results ---
# The output is a list of dictionaries. We grab the very first item [0] and extract its 'generated_text'
# Printing the final text to the terminal