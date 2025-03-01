import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import matplotlib.pyplot as plt
from langdetect import detect
from collections import Counter

model_name = "meta-llama/Llama-2-7b-chat-hf"


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # quantizare pe 4 biti
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)


tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# system prompt personalizat pentru sarcina noastra
system_prompt = (
    "<s>[INST] <<SYS>>\n"
    "You are a creative and knowledgeable linguistics expert with a deep understanding of phonetics, grammar, "
    "and language construction."
    "Your task is to invent a completely original language, complete with unique vocabulary, grammar rules, "
    "and phonetic structure."
    "This language should not borrow words or grammatical structures from existing languages, ensuring it is entirely "
    "distinct and unique. It must especially not use pronouns or verbs from English."
    "The language must include:\n"
    "- A phonetic system (sounds and how words are pronounced).\n"
    "- Basic grammar rules, including sentence structure and word order (must be Subject-Object-Verb).\n"
    "- A starting vocabulary, which should contain at least 10 nouns, 10 verbs and 10 adjectives, along with "
    "pronouns.\n"
    "- Rules for forming new words (e.g., prefixes, suffixes, or root transformations).\n\n"
    "After creating the language, you will be asked to write original texts using it. Ensure all definitions, "
    "grammar rules, and structures remain consistent."
    "Be creative and thorough, but prioritize consistency and coherence in your invented language.\n"
    "<</SYS>>\n\n"
)

log_dir = "conversation_logs"
os.makedirs(log_dir, exist_ok=True)


def get_next_conversation_id(): # generam numere pentru ID-urile conversatiilor
    files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
    ids = sorted([int(f.split(".")[0]) for f in files if f.split(".")[0].isdigit()])
    return ids[-1] + 1 if ids else 1


def load_conversation(conversation_id):
    file_path = os.path.join(log_dir, f"{conversation_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        print("Conversation not found.")
        return None


def save_conversation(conversation_id, conversation_history):
    file_path = os.path.join(log_dir, f"{conversation_id}.json")
    with open(file_path, "w") as f:
        json.dump(conversation_history, f, indent=4)


def collect_data(source):
    # map pentru a face codul cat de cat modular pentru ambele LLM-uri
    file_mapping = {
        "llama": ("text_archive_llama.txt", "llama_language_distribution_pie_chart.png"),
        "gpt": ("text_archive_gpt.txt", "gpt_language_distribution_pie_chart.png")
    }

    if source not in file_mapping:
        print("Invalid source. Please choose either 'llama' or 'gpt'.")
        return

    file_path, chart_path = file_mapping[source]
    language_counts = Counter()

    if not os.path.exists(file_path):
        print(f"{file_path} not found. Please ensure the file exists in the working directory.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # sectiunile de text din fisier sunt separate de cate 3 linii goale
    sections = content.split("\n\n\n")

    for section in sections:
        lines = section.strip().split("\n")
        text_lines = []
        collecting_text = False

        for line in lines:
            if line.startswith("Text"):
                collecting_text = True  # preluam textul de la prima linie de sub linia care incepe cu "Text"
            elif line.startswith("Translation"):
                collecting_text = False  # pana la ultimul text de deasupra liniei care incepe cu "Translation"
                break
            elif collecting_text:
                text_lines.append(line)  # check pentru text care se intinde pe mai multe linii

        if text_lines:
            combined_text = " ".join(text_lines)
            try:
                detected_language = detect(combined_text)
                language_counts[detected_language] += 1
            except Exception as e:
                print(f"Error detecting language for text: {combined_text[:30]}... Error: {e}")

    print(f"Detected language counts for {source}:")
    for lang, count in language_counts.items():
        print(f"{lang}: {count}")
    save_pie_chart(language_counts, chart_path)


def save_pie_chart(language_counts, chart_path):
    labels = language_counts.keys()
    sizes = language_counts.values()
    # setari pentru pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Language Distribution in Text Archive")
    plt.axis("equal")
    plt.savefig(chart_path)
    print(f"Pie chart saved as {chart_path}")
    plt.close()


def tokenize_and_analyze_llama():
    file_path = "text_archive_llama.txt"
    output_file = "llama_token_analysis.txt"

    if not os.path.exists(file_path):
        print(f"{file_path} not found. Please ensure the file exists in the working directory.")
        return

    token_count = 0
    word_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # sectiunile de text din fisier sunt separate de cate 3 linii goale
    sections = content.split("\n\n\n")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for section in sections:
            lines = section.strip().split("\n")
            text_lines = []
            collecting_text = False

            for line in lines:
                if line.startswith("Text"):
                    collecting_text = True  # preluam textul de la prima linie de sub linia care incepe cu "Text"
                elif line.startswith("Translation"):
                    collecting_text = False  # pana la ultimul text de deasupra liniei care incepe cu "Translation"
                    break
                elif collecting_text:
                    text_lines.append(line)  # check pentru text care se intinde pe mai multe linii

            if text_lines:
                combined_text = " ".join(text_lines)
                tokens = tokenizer.tokenize(combined_text)
                # ignoram token-urile care nu contin deloc litere/cifre(pentru numaratoarea de tokens per cuvant)
                filtered_tokens = [token for token in tokens if any(c.isalnum() for c in token)]
                token_count += len(filtered_tokens)
                word_count += len(combined_text.split())
                out_f.write(f"Text: {combined_text}\n")
                out_f.write(f"Tokens: {filtered_tokens}\n\n")

    average_tokens_per_word = token_count / word_count if word_count > 0 else 0
    print(f"Total Tokens: {token_count}")
    print(f"Total Words: {word_count}")
    print(f"Average Tokens per Word: {average_tokens_per_word:.2f}")
    with open(output_file, "a", encoding="utf-8") as out_f:
        out_f.write(f"\nTotal Tokens: {token_count}\n")
        out_f.write(f"Total Words: {word_count}\n")
        out_f.write(f"Average Tokens per Word: {average_tokens_per_word:.2f}\n")


def conversation_menu():
    while True:
        print("\nWelcome to Llama 2 chatbot!")
        print("Would you like to:")
        print("1. Start a new conversation")
        print("2. Continue a previous conversation")
        print("3. Exit")
        print("4. Initialize Data Collection and Analysis")
        print("5. Tokenize and Analyze Llama Texts")
        choice = input("Enter 1, 2, 3, 4, or 5: ")

        if choice == "1":
            conversation_id = get_next_conversation_id()
            print(f"Starting a new conversation. Your conversation ID is: {conversation_id}")
            return conversation_id, []

        elif choice == "2":
            conversation_id = input("Enter the conversation ID: ")
            try:
                conversation_id = int(conversation_id)
            except ValueError:
                print("Invalid conversation ID. Please try again.")
                continue

            conversation_history = load_conversation(conversation_id)
            if conversation_history is not None:
                print(f"Continuing conversation with ID: {conversation_id}")
                return conversation_id, conversation_history
            else:
                print("Conversation not found. Please try again.")

        elif choice == "3":
            print("Exiting. Goodbye!")
            exit()

        elif choice == "4":
            source = input("Which data source would you like to analyze? ('llama' or 'gpt'): ").strip().lower()
            collect_data(source)

        elif choice == "5":
            tokenize_and_analyze_llama()

        else:
            print("Invalid choice. Please try again.")


# Main loop
while True:
    conversation_id, conversation_history = conversation_menu()

    print("Start chatting with Llama 2 (type 'exit' to return to the menu):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Returning to the main menu. Conversation saved.")
            save_conversation(conversation_id, conversation_history)  # Save conversation on exit
            break

        # actualizam istoricul conversatiei
        conversation_history.append({"user": user_input})

        # combinam system prompt-ul mereu cu istoricul(care contine si prompt-ul nostru curent)
        prompt = system_prompt
        for entry in conversation_history:
            if "user" in entry:
                prompt += f"{entry['user']} [/INST] "
            if "bot" in entry:
                prompt += f"{entry['bot']} </s><s>[INST] "

        # generam raspuns bot
        response = text_gen_pipeline(
            prompt,
            max_length=4096,  # dupa 4096 tokeni nu functioneaza bine, deci nu are rost
            min_length=20,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
            truncation=True
        )

        # extragem si formatam textul pentru a fi citibil
        response_text = response[0]["generated_text"]
        bot_response = response_text.split("[/INST]")[-1].strip()

        # adaugam raspunsul bot-ului peste istoric
        conversation_history.append({"bot": bot_response})

        print(f"Llama: {bot_response}")

        # codul pentru tokens_debug care pana la urma nu a fost prea util

        # tokens = tokenizer.encode(bot_response)
        # tokens_text = tokenizer.convert_ids_to_tokens(tokens)
        # with open("tokens_debug.txt", "a", encoding="utf-8") as f:
        # f.write(f"Response: {bot_response}\n")
        # f.write(f"Tokens (Text): {tokens_text}\n")
        # f.write(f"Number of tokens: {len(tokens)}\n\n")
