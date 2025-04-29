# --- START OF FILE phase2_cot_fir_heuristics_refined.py ---

import os
import time
import torch
import google.generativeai as genai
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np
import re # Import regex for heuristic checks

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- Configuration ---
# *** POINT TO THE DATABASE BUILT BY THE SIMPLIFIED phase1pt9.py ***
PERSIST_DIRECTORY = 'db_bns_generalized_sentences_groq3_cot' # MUST MATCH PHASE 1 OUTPUT DIR
EMBEDDING_MODEL_NAME = 'intfloat/e5-large-v2' # MUST MATCH PHASE 1 EMBEDDING MODEL

# --- Phase 2 Specific Configuration ---
GEMINI_MODEL_NAME = 'models/gemini-2.0-flash'
load_dotenv(); GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY: exit("Error: GOOGLE_API_KEY not found.")

# Configure Gemini
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"Gemini configured: {GEMINI_MODEL_NAME}")
except Exception as e: exit(f"Error configuring Gemini: {e}")

# Search and Result Limits
SEARCH_K = 20
MAX_RESULTS_TO_SHOW = 25

# --- Boost Factors ---
KEY_ACTIONS = {
    # ... (Keep your existing KEY_ACTIONS dictionary) ...
    'abduct': 1.15, 'abduction': 1.15, 'abet': 1.05, 'abetment': 1.05, 'abett': 1.05, 'abettor': 1.05,
    'acid': 1.15, 'assault': 1.1, 'attempt': 1.05, 'banknote': 1.05, 'break': 1.15, 'build': 1.05,
    'building': 1.05, 'bullock': 1.08, 'cash': 1.05, 'cashier': 1.05, 'cheat': 1.08, 'cheatsa': 1.08,
    'child': 1.05, 'children': 1.05, 'coin': 1.05, 'confin': 1.1, 'confine': 1.08, 'confinement': 1.1,
    'conspir': 1.05, 'conspiracy': 1.05, 'conspire': 1.05, 'counterfeit': 1.1, 'criminal': 1.05,
    'currencynote': 1.05, 'dacoit': 1.15, 'dacoity': 1.15, 'damage': 1.08, 'dishonest': 1.1,
    'document': 1.05, 'documenta': 1.05, 'dwell': 1.05, 'dwelling': 1.05, 'enter': 1.1, 'explosive': 1.12,
    'extort': 1.12, 'extortion': 1.08, 'extortiona': 1.08, 'false': 1.05, 'falsely': 1.05, 'fire': 1.08,
    'forc': 1.1, 'force': 1.08, 'forg': 1.08, 'forge': 1.08, 'forgery': 1.08, 'forgerya': 1.08,
    'fraudulen': 1.08, 'fraudulently': 1.05, 'grievous': 1.12, 'homicide': 1.12, 'homicidea': 1.15,
    'house': 1.05, 'housebreak': 1.15, 'housebreakinga': 1.15, 'housetrespas': 1.15, 'hurt': 1.1,
    'hurtwhoever': 1.1, 'injur': 1.1, 'injure': 1.1, 'intent': 1.05, 'intimidat': 1.12, 'intimidate': 1.1,
    'intimidation': 1.1, 'kidnap': 1.15, 'kidnapp': 1.15, 'know': 1.05, 'knowingly': 1.05,
    'knowledge': 1.05, 'known': 1.05, 'lock': 1.08, 'malicious': 1.08, 'misappropriate': 1.08,
    'mischief': 1.08, 'mischiefa': 1.08, 'murder': 1.15, 'murdera': 1.15, 'negligen': 1.05,
    'negligence': 1.05, 'negligent': 1.05, 'negligently': 1.05, 'note': 1.05, 'organised crime': 1.18, 'organized crime': 1.18,
    'poison': 1.1, 'poisonou': 1.1, 'probable': 1.15, 'property': 1.05, 'public servant': 1.05,
    'publication': 1.05, 'publicly': 1.05, 'rape': 1.15, 'rash': 1.05, 'rashly': 1.05, 'record': 1.05,
    'restrain': 1.08, 'restraint': 1.1, 'rob': 1.15, 'robbery': 1.15, 'servant': 1.05, 'shop': 1.05,
    'snatch': 1.12, 'stamp': 1.05, 'steal': 1.15, 'sunset': 1.08, 'terrorist': 1.2, 'theft': 1.15,
    'thefta': 1.15, 'theftexplanation': 1.15, 'threat': 1.1, 'threaten': 1.1, 'trespas': 1.15,
    'trespass': 1.12, 'unlawful': 1.05, 'unlawfully': 1.05, 'vehicle': 1.05, 'vessel': 1.05,
    'weapon': 1.1, 'woman': 1.05,
    'assembly': 1.1, 'rioting': 1.15, 'mob': 1.1 # Added group terms
}

SECTION_TYPE_MODIFIERS = {
    # ... (Keep your existing SECTION_TYPE_MODIFIERS dictionary) ...
    'abduction': 1.15, 'abetment': 0.9, 'accident': 0.9, 'affray': 1.15, 'air force': 0.5,
    'application': 0.8, 'army': 0.5, 'assault': 1.0, 'assaulting': 1.15, 'attempt': 0.95,
    'breaking': 1.2, 'breaking open': 1.15, 'cheating': 0.85, 'child': 0.9, 'coin': 0.5, 'coining': 0.7,
    'commencement': 0.8, 'commutation': 0.8, 'confine': 0.8, 'confinement': 0.8, 'consent': 0.9,
    'conspiracy': 0.9, 'counterfeit': 0.8, 'counterfeiting': 1.15, 'criminal force': 1.0,
    'criminal trespass': 1.1, 'dacoity': 1.15, 'defamation': 0.75, 'defence': 0.9, 'defined': 0.8,
    'definitions': 0.8, 'documents': 0.7, 'drugs': 0.7, 'election': 0.5, 'elections': 0.7,
    'evidence': 0.8, 'explanations': 0.8, 'extortion': 1.15, 'fine': 0.8, 'force': 1.15, 'forgery': 0.85,
    'grievous': 1.2, 'homicide': 0.8, 'house trespass': 1.15, 'house-breaking': 1.2, 'housebreaking': 1.2,
    'housetrespass': 1.15, 'hurt': 1.15, 'intimidation': 1.15, 'intoxication': 0.9, 'kidnapping': 1.15,
    'liability': 0.8, 'marriage': 0.6, 'mischief': 1.15, 'murder': 0.8, 'navy': 0.5, 'negligence': 0.9,
    'nuisance': 0.7, 'public': 0.7, 'public servant': 0.95, 'publication': 0.7, 'punishment': 0.8,
    'punishments': 0.8, 'rape': 0.85, 'religion': 0.6, 'repeal': 0.8, 'rioting': 1.05, 'robbery': 1.15,
    'savings': 0.8, 'sentence': 0.8, 'sexual': 0.7, 'stamp': 0.5, 'stamps': 0.7, 'state': 0.6,
    'statement': 0.7, 'statements': 0.7, 'theft': 1.15, 'trespass': 1.15, 'trespassing': 1.15,
    'unlawful assembly': 1.05, 'unsound': 0.9,
    'insulting modesty': 1.1 # Added
}

# --- GPU Configuration ---
if torch.cuda.is_available(): DEVICE = 'cuda'; print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else: DEVICE = 'cpu'; print("WARNING: CUDA not available. Using CPU.")

# --- <<< Refined CoT FIR Analysis Function >>> ---
def get_concepts_from_gemini_cot(narrative, model_name=GEMINI_MODEL_NAME):
    """
    Analyzes FIR narrative using Gemini with a refined CoT prompt, extracts structured concepts.
    """
    # Refined CoT prompt asking for specific details relevant to tricky sections
    prompt = f"""
    You are an expert legal assistant analyzing an incident narrative (FIR). Your goal is to extract key factual elements and actions described that are potentially relevant for identifying applicable BNS sections. Do NOT suggest section numbers. Focus solely on extracting and structuring the facts.

    Follow these steps carefully and output your reasoning AND the final list:
    1.  **Reasoning:** Under the heading "## Reasoning Analysis:", briefly explain your thought process for identifying the key elements below. Specifically mention if multiple actors seem involved in a common goal or if there's evidence of an unlawful assembly or riot. Also note any actions potentially insulting modesty or intended to provoke.
    2.  **Extraction:** Under separate headings (## Actions:, ## Objects:, ## Intent:, ## Circumstances:), list the concise factual elements extracted from the narrative (3-10 words each).
        *   Under ## Actions: Specifically note any acts, words, or gestures described as insulting, obscene, intended to provoke, violating privacy, or threatening. Detail acts of group violence.
        *   Under ## Circumstances: Specifically note if multiple people acted together with a shared goal (common intention/object), if the act occurred in public causing disturbance, or if an unlawful assembly formed. Note if actions were directed at a woman's modesty.

    Narrative:
    ---
    {narrative}
    ---

    ## Reasoning Analysis:
    [Explain reasoning step-by-step here, including group action/intent/insulting modesty checks]

    ## Actions:
    - [Action 1: Detail any insults, threats, group violence]
    - [Action 2]
    ## Objects:
    - [Object 1]
    ## Intent:
    - [Intent 1: e.g., to provoke breach of peace, to insult modesty, common object of assembly]
    ## Circumstances:
    - [Circumstance 1: e.g., Multiple actors with common intention, Public place disturbance, Unlawful assembly formed, Act against woman's modesty]
    """
    try:
        print("\nAsking Gemini to decompose narrative (Refined CoT Analysis)...")
        start_time = time.time()
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.2)
        response = model.generate_content(prompt, generation_config=generation_config)
        print(f"Gemini CoT response received ({time.time() - start_time:.2f}s).")

        if not response.parts:
             print("Warning: Gemini CoT response blocked or empty.")
             if response.prompt_feedback.block_reason: print(f"Reason: {response.prompt_feedback.block_reason}")
             return None

        raw_text = response.text
        # --- Parsing CoT Output ---
        concepts = []
        actions = re.findall(r"## Actions:\s*(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
        objects = re.findall(r"## Objects:\s*(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
        intent = re.findall(r"## Intent:\s*(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
        circumstances = re.findall(r"## Circumstances:\s*(.*?)(?=\n##|\Z)", raw_text, re.DOTALL | re.IGNORECASE)

        def extract_list_items(text_block):
            items = []
            if text_block:
                found_items = re.findall(r"^\s*[-*]\s*(.*)", text_block[0], re.MULTILINE)
                items = [item.strip() for item in found_items if item.strip()]
            return items

        concepts.extend(extract_list_items(actions))
        # concepts.extend(extract_list_items(objects)) # Often less useful for section matching, keep optional
        concepts.extend(extract_list_items(intent))
        concepts.extend(extract_list_items(circumstances))

        # Fallback
        if not concepts and "## Actions:" in raw_text:
             print("W: Regex parsing failed, attempting basic line split for concepts.")
             # ... (fallback logic as before) ...
             in_section = False
             for line in raw_text.splitlines():
                 # ... (copy fallback logic from previous version if needed) ...
                 pass # Placeholder

        print(f"Extracted FIR Concepts (Refined CoT): {concepts}")
        if not concepts: print("Warning: Gemini Refined CoT extraction yielded no concepts.")
        return concepts if concepts else None

    except Exception as e:
        print(f"Error calling Gemini API (Refined CoT): {e}")
        return None
# --- <<< END Refined CoT FIR Analysis Function >>> ---

# --- Main Execution Logic ---
if __name__ == "__main__":
    print("\n--- Phase 2: Suggesting BNS Sections (Refined CoT FIR + Heuristics) ---")
    print("-" * 85)

    # 1. Initialize Embedding Model
    print(f"\nStep 1: Initializing Embedding Model '{EMBEDDING_MODEL_NAME}' on {DEVICE}...")
    start_time_embed_init = time.time()
    try:
        embedding_model = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE}, encode_kwargs={'normalize_embeddings': True}
        )
        _ = embedding_model.embed_query("test"); print(f"Embedding model OK ({time.time() - start_time_embed_init:.2f}s).")
    except Exception as e: exit(f"Error initializing embedding model: {e}")

    # 2. Load Vector Store
    print(f"\nStep 2: Loading Vector Store from '{PERSIST_DIRECTORY}'...")
    if not os.path.exists(PERSIST_DIRECTORY): exit(f"Error: Directory '{PERSIST_DIRECTORY}' not found.")
    try:
        start_time = time.time()
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
        count = vector_store._collection.count(); print(f"Vector store loaded ({time.time() - start_time:.2f}s). Found {count} principles.")
        if count == 0: print("Warning: Vector store is empty!")
    except Exception as e: exit(f"Error loading vector store: {e}")

    # 3. Get Incident Story (Using Case 8 again for rioting context)
    print("\nStep 3: Incident Narrative Input")
    incident_narrative = """
    (First Information Report):\n\nStatement dated 12/01/2025\nMrs. Ayesha Bee Moinuddin Sheikh, Age 91 years, Residing at House No. 17, Chawl No. 3, MHB Colony Gate No. 08 Malvani Malad West Mumbai, Mobile No. 8454044062\nUpon being questioned directly, I state that I reside at the above-mentioned address and I stay at home as my health is not good.\n\nToday, on 12/01/2025, my son named Munir B. Moinuddin Sheikh went to Noor Masjid at around 07:00 PM to offer Namaz (prayers). At that time, the door of my house was open. After my son left, I got up from the bed and went to the kitchen when there was no one in my house.\n\nAfter going to the kitchen, I started washing the tea utensils there. Then someone put their hand around my neck from behind. When I turned back, there was a woman who suddenly threw chili powder on my face and eyes. The woman snatched the gold chain from my neck. When I resisted her and said, \"Who are you, you bitch?\", she grabbed my neck and the chain and dragged me to the door. After reaching the door, I threw the glass in my hand at her and when I started shouting, she ran away from there. I can identify her if I see her. Her description is as follows: She was wearing a black burqa and a white mask on her face. I don't know her. My gold chain is one and a half tolas and its approximate value would be one and a half lakh rupees.\n\nSo, today on 12/01/2025, around 07:15 PM, a woman came to my house and tried to snatch the gold chain from my neck by throwing chili powder on my face. Therefore, I have a legal complaint against that unknown woman.\nMy above statement has been typed on the computer and after reading it in Marathi, it has been explained to me in Hindi, which is true and correct as per my narration.\n\nIn person\n(Vishal Raut)\nPolice Sub-Inspector\nMalvani Police Station, Mumbai\n\nLLF.-I (Integrated Investigation Form - 1)
    """
    print("Using sample FIR narrative involving multiple assailants (Case 8).")

    # 4. Decompose Narrative (Refined CoT) & Embed FIR Concepts
    print("\nStep 4: Decomposing Narrative (Refined CoT) & Embedding FIR Concepts...")
    # <<< USE THE REFINED CoT FUNCTION >>>
    fir_concepts_raw = get_concepts_from_gemini_cot(incident_narrative)
    if fir_concepts_raw is None or not fir_concepts_raw:
        exit("Exiting: Failed to get concepts from FIR narrative using Refined CoT.")

    fir_concepts_prefixed = [f"query: {concept}" for concept in fir_concepts_raw]
    try:
        fir_embeddings = embedding_model.embed_documents(fir_concepts_prefixed)
        print(f"Successfully embedded {len(fir_embeddings)} FIR concepts.")
    except Exception as e: exit(f"Error embedding FIR concepts: {e}")

    # 5. Initial Vector Search
    print(f"\nStep 5: Performing Initial Vector Search against Principles (k={SEARCH_K})...")
    all_retrieved_principles_with_scores = []
    search_start_time = time.time()
    for i, concept in enumerate(fir_concepts_raw):
        if not concept: continue
        query_embedding = fir_embeddings[i]
        try:
            retrieved = vector_store.similarity_search_by_vector_with_relevance_scores(embedding=query_embedding, k=SEARCH_K)
            all_retrieved_principles_with_scores.extend(retrieved)
        except Exception as e: print(f"W: Error during search for concept '{concept}': {e}")
    print(f"Initial vector search completed ({time.time() - search_start_time:.2f}s). Found {len(all_retrieved_principles_with_scores)} candidates.")
    if not all_retrieved_principles_with_scores: exit("Exiting: No relevant principles found.")

    # 6. Aggregate Results & Calculate Initial Boost Factors
    print("\nStep 6: Aggregating Results & Calculating Boost Factors...")
    section_ranking_data = defaultdict(lambda: {'count': 0, 'best_score': 1.0, 'metadata': None})
    aggregation_start_time = time.time()
    unique_sections_processed = set()
    for principle_doc, principle_score in all_retrieved_principles_with_scores:
        metadata = principle_doc.metadata
        if 'bns_section' in metadata:
            section_id = metadata['bns_section']
            unique_sections_processed.add(section_id)
            data = section_ranking_data[section_id]
            data['count'] += 1
            if principle_score < data['best_score']:
                data['best_score'] = principle_score
                data['metadata'] = metadata

    # Calculate initial boosts based on titles
    boost_factors = {}
    print("  Calculating initial heuristic boost factors...")
    for section_id in unique_sections_processed:
        data = section_ranking_data.get(section_id)
        boost = 1.0
        if data and data['metadata']:
            section_title = data['metadata'].get('sectionHead', '').lower()
            for action, factor in KEY_ACTIONS.items():
                if action in section_title: boost *= factor
            for type_kw, modifier in SECTION_TYPE_MODIFIERS.items():
                if type_kw in section_title: boost *= modifier
            boost_factors[section_id] = min(max(boost, 0.7), 1.5)
        else:
            boost_factors[section_id] = 1.0

    # --- Apply Heuristics for Specific Sections ---
    print("\nApplying Heuristics...")
    fir_text_lower = incident_narrative.lower()
    concepts_lower = [c.lower() for c in fir_concepts_raw]
    sections_found = list(section_ranking_data.keys()) # Get sections found by vector search

    # --- Heuristic for Rioting/Unlawful Assembly (189, 190, 191) ---
    triggered_riot_heuristic = False
    mentions_group_text = any(word in fir_text_lower for word in ['group', 'mob', 'together', 'all of them', 'several people'])
    mentions_force_text = any(word in fir_text_lower for word in ['force', 'violence', 'stone', 'stick', 'weapon', 'assault', 'beat', 'punches', 'kicks'])
    mentions_public_text = any(word in fir_text_lower for word in ['public', 'road', 'street', 'stand', 'disturbance'])
    concepts_mention_group = any(word in c for c in concepts_lower for word in ['multiple accused', 'common intention', 'all three', 'group', 'together', 'several people', 'assembly', 'riot'])

    if (mentions_group_text or concepts_mention_group) and mentions_force_text and mentions_public_text:
        print("INFO: Heuristics suggest potential Unlawful Assembly/Rioting.")
        triggered_riot_heuristic = True
        riot_sections_to_boost = {'189', '190', '191'} # Use a set for faster lookup
        boost_multiplier = 1.6 # Slightly stronger boost

        for section_id in sections_found:
             base_sec_num_match = re.search(r'\b(\d+)\b', section_id)
             if base_sec_num_match and base_sec_num_match.group(1) in riot_sections_to_boost:
                  current_boost = boost_factors.get(section_id, 1.0)
                  boost_factors[section_id] = min(current_boost * boost_multiplier, 2.5) # Increased cap slightly
                  print(f"  Boosting {section_id} (Rioting/Assembly) -> New Boost: {boost_factors[section_id]:.2f}")

    # --- Heuristic for Insulting Modesty/Provocation (352) ---
    triggered_352_heuristic = False
    mentions_insult_provoke = any(w in fir_text_lower for w in ['insult', 'vulgar', 'obscene', 'gesture', 'intrude', 'privacy', 'provoke', 'incite', 'breach of peace'])
    mentions_woman_context = any(w in fir_text_lower for w in ['woman', 'female', 'lady', 'her ', ' she ', 'modesty'])
    concepts_mention_insult = any(w in c for c in concepts_lower for w in ['insult', 'obscene', 'provoke', 'modesty', 'privacy'])

    if mentions_insult_provoke or concepts_mention_insult:
         # More targeted check: specifically if insult/provocation seems directed at a woman
         if mentions_woman_context or any("woman" in c for c in concepts_lower if "modesty" in c or "insult" in c):
              print("INFO: Heuristic suggests potential Insulting Modesty (Sec 352).")
              triggered_352_heuristic = True
              boost_multiplier = 1.3 # Moderate boost
              for section_id in sections_found:
                   if '352' in section_id:
                        current_boost = boost_factors.get(section_id, 1.0)
                        boost_factors[section_id] = min(current_boost * boost_multiplier, 2.0)
                        print(f"  Boosting {section_id} (Insult Modesty) -> New Boost: {boost_factors[section_id]:.2f}")
                        break # Boost only the first match for 352
         elif any(w in fir_text_lower for w in ['provoke', 'incite', 'breach of peace']):
              print("INFO: Heuristic suggests potential Provocation (Sec 352).")
              triggered_352_heuristic = True
              boost_multiplier = 1.2 # Slightly lower boost for general provocation
              for section_id in sections_found:
                   if '352' in section_id:
                        current_boost = boost_factors.get(section_id, 1.0)
                        boost_factors[section_id] = min(current_boost * boost_multiplier, 1.8)
                        print(f"  Boosting {section_id} (Provocation) -> New Boost: {boost_factors[section_id]:.2f}")
                        break

    print(f"\nAggregation & Boost Calculation completed ({time.time() - aggregation_start_time:.2f}s).")

    # 7. Re-rank Sections
    print("\nStep 7: Re-ranking Sections & Presenting Suggestions...")
    ranked_section_ids = sorted(
        section_ranking_data.keys(),
        key=lambda s_id: (
            section_ranking_data[s_id]['best_score'] / boost_factors.get(s_id, 1.0), # Boosted Score (Asc)
           -section_ranking_data[s_id]['count']                                      # Frequency (Desc)
        )
    )

    # --- Rule-Based Application for Section 190 & Section 3 ---
    # Check for Section 190 (Membership) if 189 is highly ranked
    suggest_190 = False
    highly_ranked_189 = False
    rank_189 = -1
    base_189_id_str = "BNS Section 189"
    for i, sec_id in enumerate(ranked_section_ids[:10]): # Check top 10
        if base_189_id_str in sec_id:
            highly_ranked_189 = True
            rank_189 = i + 1
            break
    if highly_ranked_189:
        found_190_in_list = any('190' in sec_id for sec_id in ranked_section_ids)
        if not found_190_in_list:
             suggest_190 = True
             print(f"INFO: Section 189 found at Rank {rank_189}. Adding Section 190 (Membership) as likely relevant.")

    # Check for Section 3 (Common Intention)
    suggest_sec_3 = False
    if any(word in c for c in concepts_lower for word in ['multiple accused', 'common intention', 'all three', 'group', 'together', 'several people', 'all of them', 'accomplices', 'concert']):
        if ranked_section_ids: # Check if any substantive sections were suggested
            suggest_sec_3 = True

    # 8. Present Final Suggestions
    print(f"\n--- Final Suggested BNS Sections (Ranked by Boosted Score + Frequency) ---")
    if ranked_section_ids:
        print(f"(Showing Top {MAX_RESULTS_TO_SHOW} results from {len(ranked_section_ids)} unique sections found)")
        print("-" * 170)
        print(f"{'Rank':<5} | {'Section':<18} | {'Title':<45} | {'AdjScore':<10} | {'Boost':<6} | {'Freq':<4} | {'Best Matching Principle':<70}")
        print("-" * 170)
        for i, section_id in enumerate(ranked_section_ids[:MAX_RESULTS_TO_SHOW]):
            data = section_ranking_data[section_id]
            rank = i + 1
            title = data['metadata'].get('sectionHead', 'N/A')[:45] if data['metadata'] else 'N/A'
            best_raw_score = data['best_score']
            boost = boost_factors.get(section_id, 1.0)
            adjusted_score = best_raw_score / boost
            freq = data['count']
            best_stmt = data['metadata'].get('generalized_statement', 'N/A')[:70] if data['metadata'] else 'N/A'
            print(f"{rank:<5} | {section_id:<18} | {title:<45} | {adjusted_score:<10.4f} | {boost:<6.2f} | {freq:<4} | {best_stmt:<70}")
        print("-" * 170)
        if len(ranked_section_ids) > MAX_RESULTS_TO_SHOW:
             print(f"... plus {len(ranked_section_ids) - MAX_RESULTS_TO_SHOW} more sections retrieved.")

        # Add notes for rule-based suggestions
        if suggest_190:
            print("\nNOTE: Based on the presence of Section 189 (Unlawful Assembly), consider Section 190 (Being member of unlawful assembly).")
        if suggest_sec_3:
            print("\nNOTE: Based on multiple actors, consider applying BNS Section 3 (Common Intention) alongside the relevant suggested offences.")

    else:
        print("No relevant BNS sections could be suggested based on the FIR analysis.")

    print("\nPhase 2 Completed.")
    print("-" * 85)

# --- END OF FILE phase2_cot_fir_heuristics_refined.py ---