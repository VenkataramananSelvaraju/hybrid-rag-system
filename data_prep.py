import wikipediaapi
import nltk
import json
import random
import os

# Download NLTK tokenizer data (run once)
nltk.download('punkt')
nltk.download('punkt_tab') 

# --- CONFIGURATION ---
USER_AGENT = "HybridRAG_Assignment/1.0 (contact: your_email@example.com)" # REQUIRED by Wikipedia API
MIN_WORDS = 200
CHUNK_SIZE = 300      # Target tokens per chunk (within 200-400 range)
CHUNK_OVERLAP = 50
FIXED_URLS_FILE = "fixed_urls.json"
OUTPUT_CORPUS_FILE = "rag_corpus.json"

# Initialize Wiki API
wiki = wikipediaapi.Wikipedia(
    user_agent=USER_AGENT,
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def get_wikipedia_page(page_title):
    """Fetches a single page object."""
    page = wiki.page(page_title)
    if page.exists():
        return page
    return None

def count_words(text):
    return len(text.split())

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into chunks of tokens with overlap.
    Returns a list of chunk strings.
    """
    tokens = nltk.word_tokenize(text)
    
    # If text is shorter than chunk size, return as is
    if len(tokens) <= chunk_size:
        return [" ".join(tokens)]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        
        # Move forward by stride (size - overlap)
        start += (chunk_size - overlap)
        
        # Break if we've reached the end to avoid redundant tiny chunks
        if end >= len(tokens):
            break
            
    return chunks

def generate_fixed_urls(num=200):
    """
    Generates the initial set of 200 fixed URLs.
    Run this ONCE to create your group's 'fixed_urls.json'.
    """
    print(f"Generating {num} fixed URLs...")
    fixed_titles = []
    
    # Strategy: Use categories to get diverse stable topics
    categories = ["Category:Physics", "Category:History", "Category:Computer_science", "Category:Biology"]
    
    for cat_name in categories:
        cat = wiki.page(cat_name)
        for member in cat.categorymembers.values():
            if member.ns == wikipediaapi.Namespace.MAIN and count_words(member.text) > MIN_WORDS:
                fixed_titles.append(member.title)
                if len(fixed_titles) >= num:
                    break
        if len(fixed_titles) >= num:
            break
            
    # Save to file
    with open(FIXED_URLS_FILE, 'w') as f:
        json.dump(fixed_titles, f)
    print(f"Saved {len(fixed_titles)} fixed titles to {FIXED_URLS_FILE}")
    return fixed_titles

def get_random_pages(target_count=300):
    """
    Fetches random articles until target_count is met.
    """
    print(f"Fetching {target_count} random pages...")
    pages = []
    # Wikipedia-API doesn't have a direct "random" method, so we usually 
    # use a library or just pick from a massive category, or use the standard `random` module
    # wrapping the underlying API call if needed. 
    # A simple hack for this assignment is to fetch from a generic category listing 
    # or rely on pre-selected diverse topics if random extraction is difficult.
    
    # However, to be compliant, we can try fetching 'Special:Random' concepts
    # OR (more robustly) grab a large category and sample.
    
    # SIMPLIFIED APPROACH for assignment:
    # Use a large category like "Category:Featured_articles" to ensure quality
    cat = wiki.page("Category:Featured_articles")
    all_members = list(cat.categorymembers.values())
    random.shuffle(all_members)
    
    for member in all_members:
        if member.ns == wikipediaapi.Namespace.MAIN and count_words(member.text) > MIN_WORDS:
            pages.append(member)
        
        if len(pages) >= target_count:
            break
            
    return pages

def build_corpus():
    """Main pipeline to build the dataset."""
    
    # 1. Load Fixed Set
    if not os.path.exists(FIXED_URLS_FILE):
        fixed_titles = generate_fixed_urls(200)
    else:
        with open(FIXED_URLS_FILE, 'r') as f:
            fixed_titles = json.load(f)
            
    print(f"Loaded {len(fixed_titles)} fixed titles.")
    
    # 2. Fetch Fixed Pages
    processed_pages = []
    for title in fixed_titles:
        p = get_wikipedia_page(title)
        if p: processed_pages.append(p)

    # 3. Fetch Random Set (300)
    random_pages = get_random_pages(300)
    processed_pages.extend(random_pages)
    
    print(f"Total pages to process: {len(processed_pages)}")
    
    # 4. Process and Chunk
    full_corpus = []
    chunk_id_counter = 0
    
    for page in processed_pages:
        # Create Chunks
        text_chunks = chunk_text(page.text)
        
        for chunk in text_chunks:
            record = {
                "chunk_id": chunk_id_counter,
                "source_url": page.fullurl,
                "title": page.title,
                "text_content": chunk,
                "metadata": {
                    "source": "fixed" if page.title in fixed_titles else "random"
                }
            }
            full_corpus.append(record)
            chunk_id_counter += 1
            
    # 5. Save to JSON
    with open(OUTPUT_CORPUS_FILE, 'w') as f:
        json.dump(full_corpus, f, indent=4)
        
    print(f"Successfully created {OUTPUT_CORPUS_FILE} with {len(full_corpus)} chunks.")

if __name__ == "__main__":
    build_corpus()