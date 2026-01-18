#!/usr/bin/env python
# coding: utf-8

# In[1]:


#LLM_API_KEY = "zzz your key"
#SERPAPI_KEY = "yyy your key"
#search_engine_id = "your key"


# In[2]:


# First, force removal of the wrong package and install the correct one
get_ipython().system('pip uninstall -y dotenv')
get_ipython().system('pip install -U python-dotenv')


# In[3]:


import sys
print(sys.executable)
get_ipython().system('{sys.executable} -m pip install --user --upgrade python-dotenv')


# In[4]:


# Read the .env file
from dotenv import load_dotenv
import os

load_dotenv()  # By default, looks for a file named '.env'
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
search_engine_id = os.getenv("search_engine_id")


# In[5]:


get_ipython().system('pip install openai')


# In[6]:


import requests
from bs4 import BeautifulSoup
import openai

# Function to search Google (Using SerpAPI)
def google_search(query, api_key, num_results=3):
    #search_url = f"https://serpapi.com/search?q={query}&api_key={api_key}"
    #response = requests.get(search_url)
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&num=10"
    response = requests.get(url)
    results = response.json().get("organic_results", [])
    print(results)
    return [result["link"] for result in results[:num_results]]

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs[:10]])  # First 10 paragraphs
    except Exception as e:
        return f"Failed to scrape {url}: {e}"

# Function to generate an answer using OpenAI API (Updated for v1.0+)
def generate_answer(prompt, llm_api_key):
    client = openai.OpenAI(api_key=llm_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

# Main pipeline
def answer_from_google(query, serpapi_key, llm_api_key):
    urls = google_search(query, serpapi_key)
    extracted_content = "\n\n".join([scrape_website(url) for url in urls])
    prompt = f"{query}\n\n{extracted_content}"
    return generate_answer(prompt, llm_api_key)

# Example usage
#query = "What is quantum computing for weather and climate?"
query = "EUMETNET Artificial Intelligence and Machine Learning for WEather, Climate E-AI."
answer = answer_from_google(query, SERPAPI_KEY, LLM_API_KEY)
#print(answer)


# In[7]:


from IPython.display import display, Markdown
display(Markdown(answer))


# In[8]:


import requests
from pprint import pprint


def search_google(query, api_key, search_engine_id, num_results=5):
    """
    Performs a Google Search using the Custom Search JSON API and returns structured results.

    :param query: The search term.
    :param api_key: Your Google API key.
    :param search_engine_id: Your Custom Search Engine ID.
    :param num_results: The number of search results to return.
    :return: A list of search results or an error message.
    """

    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&num={num_results}"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return []

    # Parse JSON response correctly
    data = response.json()
    results = []

    if "items" in data:
        print(f"\n🔍 Search Results for: '{query}' (Top {num_results})")
        print("=" * 60)

        for i, item in enumerate(data["items"], start=1):
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")
            snippet = item.get("snippet", "No Description")

            # Store the structured result
            results.append({"title": title, "link": link, "snippet": snippet})

            # Print in a structured way
            print(f"📌 **Result {i}:**")
            print(f"   **Title:** {title}")
            print(f"   **Link:** {link}")
            print(f"   **Snippet:** {snippet}\n")
            print("-" * 60)
    else:
        print("❌ No results found.")

    return results

# Example usage:
API_KEY = SERPAPI_KEY  # Correct Google API Key
SEARCH_ENGINE_ID = search_engine_id  # Custom Search Engine ID

query = "EUMETNET AI programme"
search_results = search_google(query, API_KEY, SEARCH_ENGINE_ID, num_results=5)

# Use `search_results` for further processing
pprint(search_results)


# In[9]:


import requests
from bs4 import BeautifulSoup
import openai

# Function to perform a Google Search and return top result URLs
def search_google(query, api_key, search_engine_id, num_results=5):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&num={num_results}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return []

    data = response.json()
    results = []

    if "items" in data:
        print(f"\n🔍 Search Results for: '{query}' (Top {num_results})")
        print("=" * 60)

        for i, item in enumerate(data["items"], start=1):
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")

            results.append(link)  # Only store the links for scraping

            # Display retrieved links
            print(f"📌 **Result {i}:** {title}")
            print(f"   🔗 {link}")
            print("-" * 60)

    return results

# Function to scrape full text from a webpage
def scrape_website(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract paragraphs
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text() for p in paragraphs[:15]])  # Extract first 15 paragraphs

        return content if content else "No content extracted."
    except Exception as e:
        return f"Failed to scrape {url}: {e}"

# Function to generate an answer using OpenAI API
def generate_answer(prompt, llm_api_key):
    client = openai.OpenAI(api_key=llm_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Main function that integrates Google Search, Web Scraping, and LLM Answering
def answer_from_google(query, api_key, search_engine_id, llm_api_key, num_results=3):
    urls = search_google(query, api_key, search_engine_id, num_results)

    extracted_content = []
    for url in urls:
        content = scrape_website(url)
        extracted_content.append(f"\n--- Source: {url} ---\n{content}")

    # Combine all extracted content for LLM input
    full_text = "\n\n".join(extracted_content)

    prompt = f"Summarize the following content and answer the question: {query}\n\n{full_text}"

    return generate_answer(prompt, llm_api_key)

# Example usage:
API_KEY = SERPAPI_KEY  # Correct Google API Key
SEARCH_ENGINE_ID = search_engine_id  # Custom Search Engine ID

query = "Tell me about EUMETNET Artificial Intelligence Programme"
answer = answer_from_google(query, API_KEY, SEARCH_ENGINE_ID, LLM_API_KEY, num_results=3)


# In[10]:


display(Markdown(answer))

