# AI-Chatbot-with-Speech-Memory

## Overview
This project is an AI-powered chatbot with speech recognition and memory capabilities. It integrates with a language model and allows users to ask questions via text or voice input. The chatbot processes uploaded PDF documents and uses Retrieval-Augmented Generation (RAG) to generate context-aware responses. Additionally, the chatbot maintains a short-term memory of previous conversations to provide more coherent responses.

## Features
- **Speech-to-Text**: Converts user speech to text input for querying the chatbot.
- **Text-to-Speech**: Converts chatbot responses to speech for an interactive experience.
- **PDF Processing**: Extracts text from uploaded PDF files for context-aware responses.
- **Memory Feature**: Stores the last three interactions for better contextual replies.
- **Retrieval-Augmented Generation (RAG)**: Uses FAISS-based embeddings to enhance responses with relevant document chunks.
- **Scrollable Chat History**: Displays previous interactions in a user-friendly UI.

## Installation

### Prerequisites
Ensure you have Python 3.7 or later installed.

### Required Libraries
Install the necessary dependencies using pip:
```bash
pip install tkinter speechrecognition transformers gtts pymupdf langchain faiss-cpu sentence-transformers numpy
```

## How It Works
1. **Run the Application**: Execute the script to launch the chatbot UI.
2. **Upload a PDF (Optional)**: Click the "Upload PDF" button to provide contextual knowledge.
3. **Ask a Question**:
   - Type in the input field and press Enter.
   - Use the "Press to Speak" button to ask a question via voice.
4. **Receive a Response**:
   - The chatbot will process the query and generate a response.
   - The response will be displayed in the chat window.
   - If enabled, the chatbot will also read the response aloud.
5. **Enable/Disable Memory**: Check the "Use Memory" option to allow the chatbot to remember the last three interactions.

## Usage Notes
- The chatbot currently uses the `meta-llama/Llama-3.2-1B-Instruct` model for responses.
- The embedding model `sentence-transformers/all-MiniLM-L6-v2` is used for document processing.
- FAISS is utilized for efficient similarity search within the document.

## Future Improvements
- Support for multiple document uploads.
- Enhanced long-term memory storage.
- UI improvements with better visualization.

## License
This project is open-source under the MIT License.

