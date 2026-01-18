#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install sentence-transformers faiss-cpu openai numpy pymupdf


# In[2]:


import sentence_transformers, transformers, huggingface_hub
print("sentence-transformers:", sentence_transformers.__version__)
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)


# In[3]:


#!git clone https://gitlab.dkrz.de/icon/icon-model.git documents


# In[4]:


import numpy as np
import os
import fitz  # PyMuPDF for PDF text extraction

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load documents from a folder recursively
def load_documents_from_folder(folder_path):
    documents = []
    file_info = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Exclude .git directories
        dirnames[:] = [d for d in dirnames if not d.startswith('.git')]

        for filename in filenames:
            if filename.startswith('.git'):
                continue  # Skip any .git files

            file_path = os.path.join(dirpath, filename)
            if filename.endswith(('.f90','.txt', '.org', '.sh', '.toml','.pdf')) \
                or '.' not in filename:  # Load .txt, .sh, .pdf, and files without extensions
                try:
                    if filename.endswith('.pdf'):
                        content = extract_text_from_pdf(file_path)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                    documents.append(content)
                    file_info.append((filename, file_path))
                    print(f"Added to DB: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    return documents, file_info


# In[5]:


folder_path = './documents'  # Replace with your folder path
documents, file_info = load_documents_from_folder(folder_path)


# In[6]:


print("Found", len(documents), "documents")


# In[7]:


from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# In[8]:


from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents and create embeddings
documents, file_info = load_documents_from_folder("documents/")
embeddings = embedder.encode(documents, convert_to_numpy=True)

# Sanity check
print("✅ Loaded documents:", len(documents))
print("🔢 Embedding shape:", embeddings.shape)
print("📄 Example file:", file_info[0])
print("📄 Example snippet:\n", documents[0][:200])


# In[9]:


print("type: ", type(embeddings))
print("shape: ", embeddings.shape)
print("MB: ", embeddings.nbytes / (1024**2))


# In[10]:


import faiss

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# In[11]:


# Query function
def query_vector_db(query, k=2):
    """
    Searches the FAISS index for the k most similar embeddings to the query.

    Parameters:
    query (str): Input text to search for similar documents.
    k (int): Number of nearest neighbors to retrieve.

    Returns:
    list of tuples: Each tuple contains (retrieved_document, file_info).
    """
    query_embedding = get_embedding(query).astype(np.float32)  # Convert to FAISS format
    distances, indices = index.search(query_embedding, k)  # Perform similarity search
    return [(documents[idx], file_info[idx]) for idx in indices[0]]


# In[12]:


def query_vector_db(query, k=5):
    # Embed the query
    query_embedding = embedder.encode([query], convert_to_numpy=True)

    # Search FAISS index
    distances, indices = index.search(query_embedding, k)

    # Return matched documents and metadata
    results = []
    for i in indices[0]:
        doc = documents[i]
        info = file_info[i] if i < len(file_info) else "No info"
        results.append((doc, info))

    return results


# In[13]:


import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# In[14]:


# Step 6: Use OpenAI to discuss results
def chat_with_openai(query,k=12):
    retrieved_docs = query_vector_db(query,k=k)
    context = "\n\n".join([f"File: {file[0]}\nContent: {doc}" for doc, file in retrieved_docs])

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # using the gpt-4o-mini model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the following documents, answer the question:\n{context}\n\nQuestion: {query}"}
        ]
    )

    answer = response.choices[0].message.content

    # Append file references to the answer
    file_references = "\n\nReferenced Files:\n" + "\n".join([f"{file[0]}: {file[1]}" for _, file in retrieved_docs])

    return answer + file_references


# In[15]:


# OpenAI Answers
query = "Tell me about the structure of the ICON model"
answer = chat_with_openai(query)

from IPython.display import display, Markdown
display(Markdown(answer))


# In[16]:


import ollama

def chat_with_ollama(query, model="mistral"):
    """Queries Ollama's local LLM (Mistral or DeepSeek) with retrieved VectorDB documents."""
    retrieved_docs = query_vector_db(query, k=20)
    context = "\n\n".join([f"File: {file[0]}\nContent: {doc}" for doc, file in retrieved_docs])

    full_prompt = f"Based on the following documents, answer the question:\n{context}\n\nQuestion: {query}"

    response = ollama.chat(model=model, messages=[{"role": "user", "content": full_prompt}])

    answer = response['message']['content']

    # Append file references
    file_references = "\n\nReferenced Files:\n" + "\n".join([f"{file[0]}: {file[1]}" for _, file in retrieved_docs])

    return answer + file_references


# In[17]:


query_text = "Tell me about the structure of the ICON model"
answer = chat_with_ollama(query_text, model="deepseek-r1:1.5b")

# Display response as formatted Markdown
display(Markdown(answer))


# In[18]:


answer = chat_with_ollama(query_text, model="mistral")

# Display response as formatted Markdown
display(Markdown(answer))


# In[19]:


import os
import shutil

def backup_and_create_results_folder(base_folder="results"):
    """Backs up existing results folder by renaming it to results_nnn and creates a new empty folder."""

    if os.path.exists(base_folder):
        counter = 1
        while os.path.exists(f"{base_folder}_{counter:03d}"):
            counter += 1
        backup_folder = f"{base_folder}_{counter:03d}"
        shutil.move(base_folder, backup_folder)
        print(f"Existing results folder backed up as: {backup_folder}")

    os.makedirs(base_folder, exist_ok=True)
    return base_folder

def copy_retrieved_documents(query, k=10, results_folder="results"):
    """Finds relevant documents using FAISS and copies them to the results folder, saving the query."""

    # Backup old results and create new folder
    results_folder = backup_and_create_results_folder(results_folder)

    # Retrieve relevant documents
    retrieved_docs = query_vector_db(query, k)

    copied_files = []

    for _, file_info in retrieved_docs:
        file_path = file_info[1]  # Assuming file_info[1] contains the file path

        if os.path.exists(file_path):
            dest_path = os.path.join(results_folder, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            copied_files.append(dest_path)
        else:
            print(f"Warning: File not found - {file_path}")

    # Save query text
    query_path = os.path.join(results_folder, "query.txt")
    with open(query_path, "w", encoding="utf-8") as f:
        f.write(query)

    print(f"Copied {len(copied_files)} files to {results_folder}")
    print(f"Query saved in {query_path}")


# In[20]:


# Example usage
query_text = "What turbulence scheme in ICON?"
copy_retrieved_documents(query_text, k=5)


# In[21]:


# OpenAI Answers
query = "What turbulence scheme in ICON?"
answer = chat_with_openai(query)

from IPython.display import display, Markdown
display(Markdown(answer))


# In[22]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
get_ipython().run_line_magic('ls', '')


# In[23]:


from PyPDF2 import PdfReader

def extract_pages_from_pdf(pdf_path):
    """Reads a PDF and returns a list of page texts."""
    reader = PdfReader(pdf_path)
    return [page.extract_text() for page in reader.pages]


# In[24]:


# Load the ICON tutorial as separate pages
tutorial_pdf = "ai_tutorial.pdf"
tutorial_pages = extract_pages_from_pdf(tutorial_pdf)

print(f"✅ Extracted {len(tutorial_pages)} pages from {tutorial_pdf}")

# Generate references like "ai_tutorial.pdf - Page 1", ...
tutorial_page_refs = [f"{tutorial_pdf} - Page {i+1}" for i in range(len(tutorial_pages))]

# Add to main document and file info lists
documents.extend(tutorial_pages)
file_info.extend([(ref, tutorial_pdf) for ref in tutorial_page_refs])


# In[25]:


tutorial_embeddings = embedder.encode(tutorial_pages, convert_to_numpy=True)


# In[26]:


print("FAISS Index Size:", index.ntotal)  # Total embeddings in FAISS
print("Documents list size:", len(documents))
print("File Info size:", len(file_info))


# In[27]:


index.add(tutorial_embeddings)


# In[28]:


print("FAISS Index Size:", index.ntotal)  # Total embeddings in FAISS
print("Documents list size:", len(documents))
print("File Info size:", len(file_info))


# In[29]:


# OpenAI Answers
query = "What NN architectures did we talk about? CNN? GNN? Transformers?"
answer = chat_with_openai(query, k=20)

from IPython.display import display, Markdown
display(Markdown(answer))


# In[30]:


answer = chat_with_ollama(query, model="deepseek-r1:1.5b")

# Display response as formatted Markdown
display(Markdown(answer))


# In[31]:


answer = chat_with_ollama(query, model="mistral")

# Display response as formatted Markdown
display(Markdown(answer))


# In[ ]:





# In[ ]:





# In[ ]:




