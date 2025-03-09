import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os
import fitz
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

document_text = ""
chunks = []
index = None
memory = []  # Store last 3 chats

# Load language model - Currently using TinyLlama for faster processing - Change it to better model for better performance
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load embedding model - To convert the pdf into context
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# UI Setup
root = tk.Tk()
root.title("AI Chatbot with Speech & Memory")

use_memory = tk.BooleanVar(value=True)  # Toggle memory feature - on or off

# UI Components
label_question = tk.Label(root, text="Your Question:", font=("Arial", 12))
label_question.pack(pady=5)

entry_question = tk.Entry(root, width=50)
entry_question.pack(pady=5)

label_response = tk.Label(root, text="Chatbot Response:", font=("Arial", 12))
label_response.pack(pady=5)

# Scrollable chat history
chat_history = scrolledtext.ScrolledText(root, height=10, width=60, state=tk.DISABLED)
chat_history.pack(pady=5)

status_label = tk.Label(root, text="", font=("Arial", 10), fg="blue")
status_label.pack(pady=5)

# Upload pdf file and extract text from it.
def upload_pdf():
    global document_text, chunks, index
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        document_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(document_text)
        index, chunks = build_faiss_index(chunks)
        print("PDF Uploaded and Processed!")

upload_button = tk.Button(root, text="Upload PDF", command=upload_pdf)
upload_button.pack(pady=10)

memory_checkbox = tk.Checkbutton(root, text="Use Memory (Last 3 Chats)", variable=use_memory)
memory_checkbox.pack(pady=5)

speak_button = tk.Button(root, text="Press to Speak", command=lambda: threading.Thread(target=speech_to_text).start())
speak_button.pack(pady=10)

# Convert the speech to text to pass it to LLM.
def speech_to_text():
    recognizer = sr.Recognizer()
    root.after(0, lambda: status_label.config(text="Listening..."))
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        root.after(0, update_question, text)
        threading.Thread(target=generate_response, args=(text,)).start()
    except sr.UnknownValueError:
        root.after(0, lambda: status_label.config(text="Could not understand. Try again."))
    except sr.RequestError:
        root.after(0, lambda: status_label.config(text="Speech service unavailable."))

# Updates the question entry in the UI
def update_question(question_text):
    entry_question.delete(0, tk.END)
    entry_question.insert(0, question_text)

# Passing data to LLM using RAG
def generate_response(query):
    root.after(0, process_response, query)

def process_response(query):
    global memory
    # Use memory if enabled
    context = "".join(memory[-3:]) if use_memory.get() else ""
    # Retrieve relevant context from the document using the embeddings
    context += retrieve_relevant_chunks(query, index, chunks)
    # Model prompt
    prompt = f"""You are an AI assistant. Answer concisely using only the given context.
                Context:
                {context}
                Question:
                {query}
                Answer:"""
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract response after 'Answer:
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    memory.append(f"Q: {query}\nA: {response}")

    # Update memory (Keep only last 3 chats)
    if len(memory) > 3:
        memory.pop(0)
    text_to_speech(response)
    root.after(0, update_chat_history, query, response)
    root.after(0, lambda: status_label.config(text=""))

# Updates the chatbot response display in the UI
def update_chat_history(question, response):
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {question}\n", "user")
    chat_history.insert(tk.END, f"Bot: {response}\n\n", "bot")
    chat_history.config(state=tk.DISABLED)
    chat_history.yview(tk.END)
    chat_history.tag_config("user", foreground="blue")
    chat_history.tag_config("bot", foreground="green")

# Converts text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("start response.mp3")  # Windows: 'start', macOS: 'afplay', Linux: 'mpg321'

# Extracts text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text("text") for page in doc)

# Splits text into chunks, can create more but due to limitation in memory less chunks are created
def chunk_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Creates a FAISS index from document chunks - For RAG embeddings
def build_faiss_index(chunks):
    embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, chunks

# Retriving the relevant chunks with respect to the user query using similarity
def retrieve_relevant_chunks(query, index, chunks, k=2):
    if index is None or not chunks:
        return ""
    query_embedding = np.array([embedding_model.embed_query(query)])
    distances, indices = index.search(query_embedding, k)
    selected_chunks = [chunks[i] for i in indices[0]]
    context = " ".join(selected_chunks[:k]) # Select top-k chunks - Reduced to 2 because of lower memory
    tokens = tokenizer(context, return_tensors="pt")['input_ids'][0]
    return tokenizer.decode(tokens[:1500], skip_special_tokens=True) if len(tokens) > 1500 else context

# Function to handle "Enter" key press to use the typed question
def on_enter(event):
    question_text = entry_question.get().strip()
    if question_text:
        threading.Thread(target=generate_response, args=(question_text,)).start()

entry_question.bind("<Return>", on_enter)

root.mainloop()
