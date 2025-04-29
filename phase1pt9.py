# --- START OF FILE phase1pt6_fixed.py ---

import json
import os
import shutil
import time
import torch
from groq import Groq, RateLimitError, APIError # Using Groq
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
import random
import nltk # For sentence tokenization
import re # Import regex for cleaning

# --- Download NLTK data if needed ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer data not found. Downloading...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")
except Exception as e:
    print(f"Warning: Error checking/downloading NLTK data: {e}")

# --- Configuration ---
BNS_JSON_PATH = 'BNS Laws.json'
# *** Directory for GROQ-GENERALIZED SENTENCE EMBEDDINGS ***
PERSIST_DIRECTORY = 'db_bns_generalized_sentences_groq3_cot' # Changed name slightly for clarity
# PERSIST_DIRECTORY = 'db_bns_generalized_sentences_groq3' # Or use the old name to overwrite

EMBEDDING_MODEL_NAME = 'intfloat/e5-large-v2'
# *** SELECT GROQ MODEL ***
# GROQ_MODEL_NAME = 'mixtral-8x7b-32768'
GROQ_MODEL_NAME = 'qwen-qwq-32b' # Try Llama3 70b for potentially better reasoning
# GROQ_MODEL_NAME = 'gemma-7b-it' # Smaller alternative

load_dotenv(); GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# *** FORCE REBUILD IS MANDATORY to overwrite with NEW CoT data ***
FORCE_REBUILD = True # MUST be True to build the new DB
# *** RATE LIMIT DELAY (Adjust based on model/tier) ***
GROQ_DELAY_SECONDS = 0.4 # Slightly increased delay for safety with CoT

# --- GPU Configuration ---
if torch.cuda.is_available(): DEVICE = 'cuda'; print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else: DEVICE = 'cpu'; print("WARNING: CUDA not available.")

# --- Configure Groq Client ---
if not GROQ_API_KEY: exit("Error: GROQ_API_KEY not found.")
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"Groq client configured successfully for model: {GROQ_MODEL_NAME}")
except Exception as e: exit(f"Error configuring Groq client: {e}")

# --- Helper Functions ---

def load_bns_data(filepath):
    # (Remains the same as before)
    if not os.path.exists(filepath): print(f"Error: JSON file not found at {filepath}"); return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        print(f"Successfully loaded {len(data)} sections from {filepath}")
        return data
    except json.JSONDecodeError as e: print(f"Error decoding JSON: {e}"); return None
    except Exception as e: print(f"Unexpected error loading JSON: {e}"); return None

# --- <<< NEW >>> REFINED Prompt Function for Sentence Generalization (Groq Format with CoT) ---
# --- <<< NEW >>> REFINED Prompt Function (Alternative Structure) ---
# --- <<< NEW >>> SIMPLIFIED Prompt Function (No CoT Output) ---
def get_sentence_generalization_messages_groq(sentence_chunk, section_id, chunk_index):
    """Creates messages list for Groq to generalize directly (No CoT output)."""

    system_prompt = """
You are an expert legal analyst specializing in the Bharatiya Nyaya Sanhita (BNS).
Your task is to analyze specific legal clauses and distill their core operational principle suitable for semantic search.
Focus on accuracy and conciseness for the final principle.
"""

    user_prompt = f"""
Analyze ONLY the following specific sentence/clause (Chunk {chunk_index+1}) from Section {section_id} of the BNS.

**Goal:** Rewrite it as a single, concise, general statement describing the core legal *action* (actus reus), required *mental state* (mens rea, e.g., intention, knowledge, dishonesty), key *circumstances*, and essential *outcome* or *condition*. Focus on the fundamental principle suitable for semantic matching against incident descriptions (FIRs).

**Exclusions:**
*   Do NOT include specific examples or illustrations (like 'A does X...').
*   Do NOT include specific punishment durations, fines, or section numbers/references (unless absolutely essential to define the *principle* itself, which is rare).
*   Do NOT include procedural details.
*   Do NOT include explanatory introductions like "The generalized principle is:".

**Output Format:**
*   Output ONLY the final generalized legal statement text.
*   If the chunk is purely illustrative, defines a term used elsewhere, or contains no discernible core legal principle/action, output only the single word "N/A".

**Sentence/Clause Text (Chunk {chunk_index+1} of Section {section_id}):**
---
{sentence_chunk}
---

**Generalized Legal Statement:**
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
# --- <<< END NEW >>> SIMPLIFIED Prompt Function ---


# --- <<< MODIFIED >>> Function to call Groq API for Generalization (Handles CoT Parsing) ---
def get_generalized_statement_from_groq(sentence_chunk, section_id, chunk_index, client, model_name, retries=3, delay_seconds=0.5):
    """
    Calls Groq API with simplified prompt, handles retries, returns direct statement.
    """
    # Basic cleaning (keep this)
    processed_chunk = re.sub(r'^\s*Illustration[s]?[:.]?\s*', '', sentence_chunk, flags=re.IGNORECASE).strip()
    processed_chunk = re.sub(r'^\s*Exception[s]?[:.]?\s*', '', processed_chunk, flags=re.IGNORECASE).strip()
    if not processed_chunk or len(processed_chunk.split()) < 4:
        return ""

    # Use the SIMPLIFIED prompt function
    try:
        messages = get_sentence_generalization_messages_groq(processed_chunk, section_id, chunk_index)
    except NameError:
         print("FATAL ERROR: `get_sentence_generalization_messages_groq` function not defined or imported.")
         return ""

    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.2, # Can slightly increase temp if needed, but start low
                max_tokens=256,  # Can be shorter now, only need the statement
                top_p=1,
                stop=None,
                stream=False
            )

            sleep_duration = delay_seconds + random.uniform(0, 0.2)
            time.sleep(sleep_duration)

            # --- <<< SIMPLIFIED PARSING >>> ---
            raw_content = chat_completion.choices[0].message.content.strip()

            # Clean potential introductions/markers the model might still add
            final_answer = re.sub(r'^[\s\-*`]+', '', raw_content) # Remove leading markers/spaces/backticks
            # Optional: Remove common introductory phrases if they appear despite instructions
            final_answer = re.sub(r'^(Here is the generalized statement:|The generalized statement is:|Generalized Statement:)\s*', '', final_answer, flags=re.IGNORECASE).strip()
            final_answer = final_answer.strip('"') # Remove surrounding quotes if added

            # --- <<< END SIMPLIFIED PARSING >>> ---

            # Check if the result is valid (not empty and not 'N/A')
            if final_answer and final_answer.lower() != "n/a":
                # print(f"  [Debug Groq Gen Final] Sec {section_id} Chunk {chunk_index+1} -> {final_answer}")
                return final_answer
            else:
                # print(f"  [Debug Groq Gen Final] Sec {section_id} Chunk {chunk_index+1} -> N/A or Empty")
                return ""

        except RateLimitError as e:
            wait_time = 7 * (attempt + 1)
            print(f"W: Groq Rate Limit Sec {section_id}/Chunk {chunk_index+1} (Att {attempt + 1}/{retries}). Waiting {wait_time}s...")
            time.sleep(wait_time)
        except APIError as e:
            if "context_length_exceeded" in str(e):
                 print(f"E: Groq Context Length Error Sec {section_id}/Chunk {chunk_index+1} (Att {attempt + 1}/{retries}): {e}. Skipping chunk.")
                 return ""
            elif "Internal Server Error" in str(e) or "Bad Gateway" in str(e):
                 print(f"W: Groq Server Error (5xx) Sec {section_id}/Chunk {chunk_index+1} (Att {attempt + 1}/{retries}): {e}. Retrying...")
                 time.sleep(5 * (attempt + 1))
            else:
                 print(f"W: Groq API Error Sec {section_id}/Chunk {chunk_index+1} (Att {attempt + 1}/{retries}): {e}")
                 if attempt < retries - 1: time.sleep(3 * (attempt + 1))
        except Exception as e:
            print(f"W: Unexpected Error during Groq call Sec {section_id}/Chunk {chunk_index+1} (Att {attempt + 1}/{retries}): {type(e).__name__} - {e}")
            if attempt < retries - 1: time.sleep(3 * (attempt + 1))

    # If loop completes without returning successfully
    print(f"E: Failed to get valid generalization for Sec {section_id}/Chunk {chunk_index+1} after {retries} attempts.")
    return ""
# --- <<< END REVISED >>> Function ---


# --- Create Generalized Chunk Documents Function (Uses Modified Groq Func) ---
def create_generalized_chunk_documents(bns_data, groq_client, groq_model_name):
    """Chunks sections, calls Groq (with CoT) to generalize, creates Documents based on generalizations."""
    documents = []
    total_sections = len(bns_data)
    print(f"\nStarting chunking & Groq CoT generalization for {total_sections} sections...")
    overall_start_time = time.time()
    total_chunks_processed = 0
    total_chunks_skipped = 0
    total_docs_created = 0
    llm_errors = 0 # Errors after retries

    if not isinstance(bns_data, list): print("Error: Expected list."); return None
    required_keys = ["sectionID", "sectionText", "sectionHead"]

    for i, section in enumerate(bns_data):
        # Progress printing
        if (i + 1) % 10 == 0 or i == total_sections - 1: # Print more frequently
             elapsed_time = time.time() - overall_start_time
             rate = (total_chunks_processed / elapsed_time) if elapsed_time > 0 else 0 # Chunks per second
             print(f"  Processing Section {i + 1}/{total_sections}... (Chunks: {total_chunks_processed}, Skipped: {total_chunks_skipped}, Created: {total_docs_created}, Errors: {llm_errors}) (~{rate:.1f} chunks/sec) (Elapsed: {elapsed_time:.1f}s)")

        if not isinstance(section, dict): continue
        if not all(key in section for key in required_keys): continue

        section_text = section.get("sectionText", "")
        section_id = section.get("sectionID", "N/A")
        section_head = section.get("sectionHead", "N/A")

        if not isinstance(section_text, str) or not section_text.strip(): continue

        # --- Chunking using NLTK (remains same) ---
        try:
            # Pre-process slightly: replace newline chars that might break NLTK
            clean_text = section_text.replace('\n', ' ').replace('\r', ' ')
            sentences = nltk.sent_tokenize(clean_text)
        except Exception as e:
            print(f"W: NLTK failed Sec {section_id}: {e}. Treating as one chunk.")
            sentences = [section_text] # Fallback

        # --- Process Each Chunk using MODIFIED Groq function ---
        for chunk_idx, sentence_chunk in enumerate(sentences):
            total_chunks_processed += 1
            generalized_statement = get_generalized_statement_from_groq(
                sentence_chunk, section_id, chunk_idx, groq_client, groq_model_name, delay_seconds=GROQ_DELAY_SECONDS
            )

            if generalized_statement: # Only create doc if statement is valid and not empty/N/A
                total_docs_created += 1
                # Add the required prefix for e5 models
                page_content = f"passage: {generalized_statement}" # Embed the generalized statement + prefix

                # Ensure metadata values are simple types (str, int, float, bool)
                metadata = {
                    "bns_section": f"BNS Section {section_id}",
                    "source_section_id": str(section_id), # Ensure string
                    "sectionHead": str(section_head), # Ensure string
                    "chunk_index": int(chunk_idx), # Ensure int
                    "generalized_statement": str(generalized_statement) # Store the actual statement
                }
                # Add optional chapter info if available and simple
                if "chapterTitle" in section and isinstance(section["chapterTitle"], str):
                     metadata["chapterTitle"] = section["chapterTitle"]
                if "chapterID" in section and isinstance(section["chapterID"], (str, int, float)):
                     metadata["chapterID"] = str(section["chapterID"]) # Ensure string

                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
            elif generalized_statement == "": # Empty string indicates skipped or N/A
                total_chunks_skipped += 1
            # If generalized_statement is None or some other error indicator (though we return "" now)
            # It's implicitly handled by the check `if generalized_statement:`

    # --- Final Summary ---
    print(f"\nFinished processing sections.")
    print(f"  Total sections iterated: {total_sections}")
    print(f"  Total sentence chunks processed: {total_chunks_processed}")
    print(f"  Chunks skipped (short/illustrative/N/A): {total_chunks_skipped}")
    print(f"  LLM generalization errors (after retries): {llm_errors}") # Note: Logic counts retried errors in `get_generalized_statement_from_groq` but doesn't bubble up count perfectly - focus on created docs.
    print(f"  Generalized statement Documents created: {total_docs_created}")
    if not documents: print("\n*** Error: No valid generalized documents were created. Check Groq responses/parsing. ***"); return None
    print(f"Successfully created {len(documents)} Documents based on generalized statements.")
    return documents

# --- Main Execution Logic ---
if __name__ == "__main__":
    print(f"--- Phase 1: Building BNS Generalized Sentence Embeddings (Groq + CoT) ---")
    print(f"Using Embedding Model: {EMBEDDING_MODEL_NAME} on {DEVICE}")
    print(f"Using LLM for generalization: Groq / {GROQ_MODEL_NAME}")
    print(f"Target Database Directory: {PERSIST_DIRECTORY}")
    print(f"Force Rebuild: {FORCE_REBUILD}")
    print("-" * 75)

    # Step 1: Load Data
    print(f"\nStep 1: Loading BNS data from '{BNS_JSON_PATH}'...")
    bns_data = load_bns_data(BNS_JSON_PATH)
    if bns_data is None: exit("Exiting: Failed to load BNS data.")

    # Step 2: Create Generalized Sentence Documents (Calls MODIFIED Groq func)
    print("\nStep 2: Chunking Sections & Generating CoT-Driven Generalized Statements via Groq LLM...")
    start_time_docs = time.time()
    generalized_documents = create_generalized_chunk_documents(bns_data, groq_client, GROQ_MODEL_NAME)
    if generalized_documents is None or not generalized_documents:
        exit("Exiting: Failed to create generalized documents. Check logs for errors.")
    print(f"Document creation & LLM generalization took {(time.time() - start_time_docs)/60:.2f} minutes.")

    # Step 3: Initialize Embeddings
    print(f"\nStep 3: Initializing Embedding Model '{EMBEDDING_MODEL_NAME}'...")
    start_time_embed_init = time.time()
    try:
        # Use 'auto' for device selection if preferred, otherwise stick to DEVICE
        embedding_model = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True} # Important for cosine similarity
        )
        # Simple test query to ensure model loads
        _ = embedding_model.embed_query("test query")
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' initialized successfully on {DEVICE}.")
        print(f"Init took {time.time() - start_time_embed_init:.2f}s.")
    except Exception as e:
        exit(f"Error initializing embedding model: {e}")

    # Step 4: Build Vector Store (for Generalized Sentences)
    vector_store = None
    start_time_db = time.time()
    db_operation = "Load/Create"

    # Force rebuild logic (Mandatory for this new data)
    if FORCE_REBUILD and os.path.exists(PERSIST_DIRECTORY):
        print(f"\nFORCE_REBUILD=True. Removing existing DB: '{PERSIST_DIRECTORY}'")
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            print("Previous directory removed.")
        except OSError as e:
            print(f"Warning: Could not remove directory '{PERSIST_DIRECTORY}': {e}. Attempting to overwrite.")
            # If removal fails, Chroma might still overwrite, but it's cleaner to remove.
    elif not FORCE_REBUILD:
         print("\n*** WARNING: FORCE_REBUILD is set to False. ***")
         print("This script generates NEW data using Chain-of-Thought.")
         print(f"It will attempt to load from '{PERSIST_DIRECTORY}', which likely contains OLD data.")
         print("Set FORCE_REBUILD = True to build the database with the new CoT generalizations.")
         # Allow loading if user insists, but it's probably not what they want
         # exit("Exiting: Set FORCE_REBUILD = True to build the database with new CoT data.")

    # Proceed with DB creation or loading
    if not os.path.exists(PERSIST_DIRECTORY) or FORCE_REBUILD:
        print(f"\nStep 4: Building NEW Vector Store with CoT Generalized Statements in '{PERSIST_DIRECTORY}'...")
        if not generalized_documents: # Should have exited earlier, but double-check
             exit("Error: No generalized documents available to build the vector store.")
        print(f"Embedding {len(generalized_documents)} CoT generalized statements using {DEVICE}...")
        db_operation = "Creation/Rebuild"
        try:
            vector_store = Chroma.from_documents(
                documents=generalized_documents, # Use the NEW documents
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
            # Persist explicitly, though from_documents usually does. Belt and suspenders.
            vector_store.persist()
            print(f"\nVector store ({db_operation}) successful.")
        except Exception as e:
            # Provide more specific error feedback if possible
            if 'CUDA out of memory' in str(e):
                 exit(f"FATAL ERROR: CUDA out of memory during embedding. Reduce batch size (if applicable in underlying library) or use CPU. Error: {e}")
            elif 'Expected metadata value to be a str, int, float or bool' in str(e):
                 print("\n--- METADATA TYPE ERROR ---")
                 print("ChromaDB requires metadata values to be simple types (string, number, boolean).")
                 print("Check the 'create_generalized_chunk_documents' function ensures all metadata values are converted correctly (e.g., using str(), int()).")
                 exit(f"Metadata type issue: {e}")
            else:
                 exit(f"Error during vector store {db_operation}: {e}")
    else:
        # This block runs only if FORCE_REBUILD is False and dir exists
        print(f"\nStep 4: Loading EXISTING Vector Store from '{PERSIST_DIRECTORY}' (FORCE_REBUILD=False)...")
        db_operation = "Loading Existing"
        try:
            vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
            count = vector_store._collection.count()
            print(f"Existing vector store loaded. Found {count} documents.")
            if count == 0:
                print("Warning: Loaded vector store is empty.")
        except Exception as e:
            exit(f"Error loading existing vector store from '{PERSIST_DIRECTORY}': {e}. Did the previous build fail? Try setting FORCE_REBUILD=True.")

    db_elapsed = time.time() - start_time_db
    print(f"Database {db_operation} took {db_elapsed:.2f} seconds.")

    # Step 5: Optional Test Search (Crucial to verify new data structure)
    print("\nStep 5: Performing Optional Test Search on NEW CoT Generalized Statements...")
    if vector_store and vector_store._collection.count() > 0:
        try:
            test_query_text = "causing grievous hurt with dangerous weapon"
            # Add the REQUIRED query prefix for E5 models
            test_query_prefixed = f"query: {test_query_text}"
            print(f"Running test query: '{test_query_prefixed}'")
            # Request score for relevance assessment
            results = vector_store.similarity_search_with_relevance_scores(test_query_prefixed, k=3) # Get a few results
            if results:
                print(f"Test search successful. Found {len(results)} result(s):")
                for i, (doc, score) in enumerate(results):
                    print(f"  Result {i+1} (Score: {score:.4f}):") # Score interpretation depends on normalization & distance metric
                    print(f"    Metadata: {doc.metadata}") # CHECK METADATA IS CORRECT
                    # Verify the generalized statement looks like a principle derived via CoT
                    print(f"    Generalized Stmt: {doc.page_content.replace('passage: ', '')}") # REMOVE passage: prefix for readability
            else:
                print("Test search ran but found 0 results. Check query or DB content.")
        except Exception as e:
            print(f"Warning: Test search failed: {e}")
    elif vector_store:
        print("Skipping test: Vector store is empty.")
    else:
        print("Skipping test: Vector store not initialized.")


    print(f"\n--- Phase 1 (CoT Generalized Sentences - Groq) Completed ---")
    print(f"Vector store for CoT generalizations REBUILT at: '{PERSIST_DIRECTORY}'")
    final_count = vector_store._collection.count() if vector_store else 'N/A'
    print(f"Contains {final_count} CoT-derived generalized BNS chunk embeddings.")
    print("\nNext Step: Ensure `phase2pt5_improved.py` uses the NEW prompt for FIR analysis (with CoT)")
    print(f"           AND points its `PERSIST_DIRECTORY` to '{PERSIST_DIRECTORY}'.")
    print("-" * 75)

# --- END OF FILE phase1pt6_fixed.py ---