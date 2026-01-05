import streamlit as st
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pdfplumber
import pandas as pd
from docx import Document
import json
import io
import os
from dotenv import load_dotenv
import requests
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import random
import sys
from collections import defaultdict

# Fix for Unicode encoding issues on Windows console
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Define API keys with placeholders
import streamlit as st

OPENROUTER_API_KEYS = {
    "Account A": st.secrets["OPENROUTER_API_KEY_A"],
    "Account B": st.secrets["OPENROUTER_API_KEY_B"]
}



# ---------- LOGGING SETUP ----------
def setup_logging():
    """Create logs directory and setup comprehensive logging with UTF-8 encoding."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"sara_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear any existing handlers
    logging.getLogger().handlers = []
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # Added encoding='utf-8'
            logging.StreamHandler()  # StreamHandler now works with reconfigured sys.stdout
        ]
    )
    
    return logging.getLogger(__name__), log_file

logger, log_file = setup_logging()

# ---------- DUAL API CONFIGURATION ----------

# Validate API keys on startup
if not OPENROUTER_API_KEYS["Account A"] or not OPENROUTER_API_KEYS["Account B"]:
    st.error("""
    ❌ API KEYS NOT CONFIGURED!
    
    Please add to your .env file:
    OPENROUTER_API_KEY_A=sk-or-v1-your-first-key
    OPENROUTER_API_KEY_B=sk-or-v1-your-second-key
    
    Then restart the app.
    """)
    st.stop()

# Updated with top free models from OpenRouter 2025, focusing on large context windows
OPENROUTER_MODELS = {
    "Xiaomi MiMo-V2-Flash (Free)": "xiaomi/mimo-v2-flash:free",  # 262K tokens, top for reasoning/extraction
    "Mistral Devstral 2 (Free)": "mistral/devstral-2:free",  # 262K, agentic coding
    "Kwaipilot KAT-Coder-Pro (Free)": "kwaipilot/kat-coder-pro-v1:free",  # 256K, tool-use
    "NVIDIA Nemotron 3 Nano (Free)": "nvidia/nemotron-3-nano:free",  # 256K, efficient extraction
    "Qwen3 Coder (Free)": "qwen/qwen3-coder:free",  # 262K, multi-step reasoning
    "Meta Llama 4 (Free)": "meta-llama/llama-4:free",  # Up to 10M extendable, versatile
}

DEFAULT_MODEL = "mistralai/devstral-2512:free"  # Best free with large tokens for extraction

# Batched logging queue
log_queue = []

def write_logs_batch():
    """Write all queued logs at once after processing."""
    if not log_queue:
        return
    
    with open(log_file, 'a', encoding='utf-8') as f:  # Ensure UTF-8 for file write
        f.write(f"\n{'='*100}\n")
        f.write(f"BATCH LOG - {datetime.now()}\n")
        f.write(f"{'='*100}\n\n")
        
        for entry in log_queue:
            if 'error' in entry:
                f.write(f"❌ ERROR - {entry['chunk']}: {entry.get('error', 'Unknown')}\n")
            elif 'tasks' in entry:
                f.write(f"✅ {entry['chunk']} ({entry['account']}): "
                       f"{entry['tasks']} tasks, {entry['deliverables']} deliverables\n")
            else:
                f.write(f"📤 {entry['chunk']} -> {entry['account']} "
                       f"(prompt: {entry['prompt_len']} chars, "
                       f"context: {entry['has_context']})\n")
    
    log_queue.clear()

# ---------- FILE READING ----------
def read_file(file_path: Path | io.BytesIO, file_extension: str) -> str:
    """Read file content with page markers for PDFs."""
    try:
        if file_extension.lower() == ".txt":
            if isinstance(file_path, io.BytesIO):
                return file_path.read().decode("utf-8")
            return file_path.read_text(encoding="utf-8")
        
        elif file_extension.lower() == ".pdf":
            full_text = ""
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages):
                    full_text += f"\n--- PAGE {page_num + 1} / {total_pages} ---\n"
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                    
                    tables = page.extract_tables()
                    if tables:
                        full_text += "\n[TABLE_START]\n"
                        for table in tables:
                            for row in table or []:
                                if row and any(cell and cell.strip() for cell in row):
                                    row_text = " | ".join(
                                        cell.strip().replace("\n", " ") if cell else "[EMPTY]"
                                        for cell in row
                                    )
                                    full_text += f"{row_text}\n"
                        full_text += "[TABLE_END]\n"
            
            combined_text = full_text.strip().replace("\r", "")
            combined_text = combined_text.encode('ascii', 'ignore').decode('ascii')
            return combined_text
        
        elif file_extension.lower() == ".docx":
            doc = Document(file_path)
            text_lines = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(text_lines)
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        logger.error(f"File read error: {e}")
        return ""

def clean_text(text: str) -> str:
    """
    Clean text by removing irrelevant sections, normalizing dashes, and removing page numbers.
    """
    if not text:
        return ""
 
    # Remove content after certain sections (from analyzed code)
    match = re.search(r"(Bibliography|Staffing Requirements|Transition Planning|Contract Deliverables|KEY PERSONNEL AND SKILLS)", text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
 
    # Normalize dashes
    text = text.replace("–", "-").replace("—", "-")
 
    # Remove page numbers (from analyzed code)
    text = re.sub(r'Page\s*\d+\s*(?:of\s*\d+)?', '', text, flags=re.IGNORECASE)
 
    # Normalize spaces and preserve newlines (from analyzed code)
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    cleaned_text = '\n'.join(line for line in cleaned_lines if line)
 
    return cleaned_text

# ---------- OPTIMIZED CHUNKING ----------
def split_into_page_chunks(text: str, pages_per_chunk: int = 6) -> List[str]:
    """
    Robust chunking:
    - Try to count page markers like '--- PAGE' (case-insensitive)
    - If no page markers found, fallback to character-based chunks (cloud-safe)
    """
    # count page markers with regex (more robust than simple count)
    total_pages = len(re.findall(r'---\s*PAGE\b', text, flags=re.IGNORECASE))

    # If no page markers, fall back to char-based chunks
    if total_pages == 0:
        # chunk by characters to keep prompt size reasonable
        max_chars = 12000
        chunks = [ text[i:i+max_chars] for i in range(0, len(text), max_chars) ]
        logger.info(f"No page markers detected. Falling back to character-based chunking: {len(chunks)} chunks")
        logger.info(f"📄 DOCUMENT SPLIT (FALLBACK) | Total Chars: {len(text)} | Chunks: {len(chunks)}")
        return chunks

    # otherwise, perform page-based chunking using the page markers
    lines = text.split('\n')
    chunks = []
    current_chunk_lines = []
    page_count = 0

    for line in lines:
        current_chunk_lines.append(line)
        if re.search(r'---\s*PAGE\b', line, flags=re.IGNORECASE):
            page_count += 1
            if page_count >= pages_per_chunk:
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = []
                page_count = 0

    if current_chunk_lines:
        chunks.append('\n'.join(current_chunk_lines))

    logger.info(f"\n{'='*100}")
    logger.info(f"📄 DOCUMENT SPLIT (OPTIMIZED)")
    logger.info(f"   Total Pages: {total_pages}")
    logger.info(f"   Pages per Chunk: {pages_per_chunk}")
    logger.info(f"   Total Chunks: {len(chunks)}")
    logger.info(f"{'='*100}\n")

    return chunks


# ---------- SMART CONTEXT EXTRACTION ----------
def extract_smart_context(prev_result: Dict, prev_chunk: str) -> str:
    """
    OPTIMIZED: Extract smart context instead of full 500 chars.
    Only include the essential information for continuity.
    """
    context_parts = []
    
    # Last task from previous result
    if prev_result.get("Tasks"):
        last_task = prev_result["Tasks"][-1]
        context_parts.append(
            f"Last Task: {last_task.get('Task', '')} "
            f"({last_task.get('Parent Task', '')})"
        )
    
    # Last deliverable
    if prev_result.get("Deliverables"):
        last_deliv = prev_result["Deliverables"][-1]
        context_parts.append(
            f"Last Deliverable: {last_deliv.get('Deliverable', '')}"
        )
    
    # Only last 200 chars of chunk (not 500)
    if prev_chunk:
        context_parts.append(f"...{prev_chunk[-200:].strip()}")
    
    return " | ".join(context_parts)
    


# ---------- STRICT REGEX PATTERNS FOR TASKS AND DELIVERABLES ----------
TASK_PATTERNS = [
    r'^\s*\d+\.\d*\s*(Task|Subtask|Phase|Activity|Milestone|Workstream|Step)\b.{20,}',  # Numbered with keyword, min 20 chars
    r'^\s*Task\s+\d+[:\-]\s*[A-Za-z].{20,}',
    r'^\s*Subtask\s+\d+[.a-zA-Z]*\s*[A-Za-z].{20,}',
    r'^\s*Phase\s+\d+[:\-]\s*[A-Za-z].{20,}',
    r'^\s*Activity\s+\d+[:\-]\s*[A-Za-z].{20,}',
    r'^\s*Workstream\s+\d+[:\-]\s*[A-Za-z].{20,}',
    r'^\s*Milestone\s+[A-Z0-9]+[:\-]\s*[A-Za-z].{20,}',
    r'^\s*•\s*(Task|Subtask|Phase|Activity|Milestone|Workstream|Step)\b.{20,}',
    r'^\s*-\s*(Task|Subtask|Phase|Activity|Milestone|Workstream|Step)\b.{20,}',
    r'(?i)^\s*(task|subtask|phase|activity|milestone|workstream|step)\s+\d+[\.:-]\s+[A-Za-z].{20,}',
    r'^\s*(Scope of Work|Performance Requirements|Task Areas|Objectives|Requirements)\s*:?\s*$',  # Headings, extract following lines
    r'^\s*\d+\.\s+(Task|Subtask|Phase|Activity|Milestone|Workstream|Step|Scope|Objective|Requirement)\b.{20,}',
    r'^\s*[A-Z]\.\s+(Task|Subtask|Phase|Activity|Milestone|Workstream|Step)\b.{20,}',
    r'^\s*[IVXLCDM]+\.\s+(Task|Subtask|Phase|Activity|Milestone|Workstream|Step)\b.{20,}',
    r'^\s*\[\d+(?:\.\d+)?\]\s+(Task|Subtask|Phase|Activity|Milestone|Workstream|Step)\b.{20,}'
]

DELIVERABLE_PATTERNS = [
    r'^\s*\d+\.\d*\s*(Deliverable|Report|Plan|Document|Training|Meeting|Assessment|Analysis)\b.{20,}',
    r'^\s*Deliverable\s+\d+[:\-]\s*[A-Za-z].{20,}',
    r'^(Kick-Off Meeting|Meeting Minutes|Weekly Status Report|Monthly Status Report|Task Schedule/Work Breakdown Structure|Quality Control Plan|Final Report|Training Materials|Implementation Plan)',
    r'^\s*•\s*(Deliverable|Report|Plan|Document|Training|Meeting|Assessment|Analysis)\b.{20,}',
    r'^\s*-\s*(Deliverable|Report|Plan|Document|Training|Meeting|Assessment|Analysis)\b.{20,}',
    r'(?i)^\s*(deliverable|report|plan|document|training|meeting|assessment|analysis)\s+\d+[\.:-]\s+[A-Za-z].{20,}',
    r'ITEM NO\s+SUPPLIES/SERVICES\s+QUANTITY\s+UNIT\s+UNIT PRICE\s+AMOUNT',  # Table headers
    r'^\s*(Deliverables|Milestones|Reports|Plans|Documents|Training|Meetings|Assessments|Analyses)\s*:?\s*$',  # Headings
    r'^\s*\d+\.\s+(Deliverable|Report|Plan|Document|Training|Meeting|Assessment|Analysis)\b.{20,}',
    r'^\s*[A-Z]\.\s+(Deliverable|Report|Plan|Document|Training|Meeting|Assessment|Analysis)\b.{20,}'
]

# Compile patterns once for efficiency
COMPILED_TASK_PATTERNS = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in TASK_PATTERNS]
COMPILED_DELIVERABLE_PATTERNS = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in DELIVERABLE_PATTERNS]

def filter_with_regex(text: str) -> str:
    """Filter text to extract candidate task/deliverable lines using strict regex, labeled separately."""
    task_candidates = set()
    deliverable_candidates = set()
    
    # Match tasks
    for pattern in COMPILED_TASK_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0).strip()
            if len(candidate) > 20:
                task_candidates.add(candidate)
    
    # Match deliverables
    for pattern in COMPILED_DELIVERABLE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0).strip()
            if len(candidate) > 20:
                deliverable_candidates.add(candidate)
    
    # Combine with labels
    labeled = []
    if task_candidates:
        labeled.append("TASK CANDIDATES:")
        labeled.extend(sorted(task_candidates))
    if deliverable_candidates:
        labeled.append("\nDELIVERABLE CANDIDATES:")
        labeled.extend(sorted(deliverable_candidates))
    
    filtered = '\n'.join(labeled)
    if not filtered:
        logger.warning("No regex matches found; using full text as fallback.")
        return text
    logger.info(f"Filtered {len(task_candidates)} task candidates and {len(deliverable_candidates)} deliverable candidates with regex.")
    return filtered

# ---------- OPTIMIZED API EXTRACTION WITH BACKOFF ----------
def retry_with_exponential_backoff(func, initial_delay=1, max_retries=6, errors=(requests.exceptions.RequestException,)):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except errors as e:
                if attempt == max_retries - 1:
                    raise
                jitter = random.random() * delay
                time.sleep(delay + jitter)
                delay *= 2  # Exponential increase
                logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
    return wrapper

@retry_with_exponential_backoff
def make_api_call(url, headers, json_data, timeout):
    return requests.post(url, headers=headers, json=json_data, timeout=timeout)

def extract_with_api(text_chunk: str, chunk_name: str, api_account: str, api_key: str, 
                     model: str, context_from_previous: str = "") -> Dict:
    """
    Filter with regex first, then extract contractually binding tasks/deliverables.
    """

    # ✅ Always initialize prompt (prevents UnboundLocalError)
    prompt = ""

    # Apply regex filter
    filtered_chunk = filter_with_regex(text_chunk)

    if not filtered_chunk.strip():
        logger.warning(f"No regex matches found in {chunk_name}; using full text fallback.")
        return {"Tasks": [], "Deliverables": []}

    # Build context section safely
    context_section = ""
    if context_from_previous:
        context_section = f"""
CONTEXT FROM PREVIOUS CHUNK (IMPORTANT - for continuity):
{context_from_previous}

---
"""

    # ✅ Build prompt OUTSIDE the if-block (CRITICAL FIX)
    prompt = f"""{context_section}
You are a senior government contract and proposal compliance analyst.

Your task is to extract ONLY CONTRACTUALLY BINDING TASKS and DELIVERABLES
from U.S. Government RFP/RFQ/SOW text.

IMPORTANT (DO NOT IGNORE):
- Government contracts encode obligations in narrative form.
- Approvals, conditions, SLAs, reporting, documentation, and compliance
  are part of the TASK and must be captured.
- Do NOT over-summarize.
- Do NOT infer tools or methodologies unless explicitly stated.
- If something is conditional (e.g., "if directed by COR"), it MUST appear in the summary.

--------------------------------
TASK EXTRACTION
--------------------------------
For each TASK:

- Task:
  Clear action-oriented task name (6–14 words allowed for precision)

- Parent Task:
  Exact section or subsection name (e.g., "5.1.7 Weekly Status Report")

- Methodology:
  Agile SDLC / ITIL / Custom / Not Specified
  (ONLY if explicitly stated; otherwise Not Specified)

- Tools & Technologies:
  ONLY tools, platforms, systems, or standards explicitly named.
  If none are stated, return "Not specified".

- Task Summary:
  Write 3–5 sentences including:
    • Contractor responsibility
    • COR / COR Designee approvals
    • Conditions or triggers
    • Frequency / SLAs / timelines
    • Reporting, documentation, governance

--------------------------------
DELIVERABLE EXTRACTION
--------------------------------
For each DELIVERABLE:

- Deliverable:
  Specific artifact name (6–14 words)

- Parent Task:
  Exact section or subsection name

- Description:
  2–4 sentences including submission method, timing,
  approval or acceptance criteria if stated

--------------------------------
OUTPUT RULES (STRICT)
--------------------------------
- Return ONLY valid JSON
- No markdown, no explanations
- No assumptions or inferred data
- Use "Not specified" when information is missing

Return JSON in EXACTLY this structure:

{{
  "Tasks": [
    {{
      "Task": "...",
      "Parent Task": "...",
      "Methodology": "...",
      "Tools & Technologies": "...",
      "Task Summary": "..."
    }}
  ],
  "Deliverables": [
    {{
      "Deliverable": "...",
      "Parent Task": "...",
      "Description": "..."
    }}
  ]
}}

CANDIDATE LINES:
{filtered_chunk}
"""

    # Queue log safely (prompt is guaranteed to exist)
    log_queue.append({
        'time': time.time(),
        'chunk': chunk_name,
        'account': api_account,
        'prompt_len': len(prompt),
        'has_context': bool(context_from_previous)
    })

    try:
        response = make_api_call(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json_data={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.35,
                "max_tokens": 4000
            },
            timeout=120
        )

        if response.status_code != 200:
            log_queue.append({
                'time': time.time(),
                'chunk': chunk_name,
                'error': f"Status {response.status_code} - {response.text[:300]}"
            })
            return {"Tasks": [], "Deliverables": []}

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        parsed = fast_parse_json(content)

        log_queue.append({
            'time': time.time(),
            'chunk': chunk_name,
            'account': api_account,
            'tasks': len(parsed["Tasks"]),
            'deliverables': len(parsed["Deliverables"])
        })

        return parsed

    except Exception as e:
        log_queue.append({
            'time': time.time(),
            'chunk': chunk_name,
            'error': str(e)
        })
        return {"Tasks": [], "Deliverables": []}


# ---------- FAST JSON PARSING ----------
def fast_parse_json(content: str) -> Dict:
    """Optimized JSON extraction with fallbacks."""
    
    # Try 1: Direct parse (fastest)
    try:
        parsed = json.loads(content)
        return ensure_structure(parsed)
    except:
        pass
    
    # Try 2: Remove markdown code blocks
    content_clean = re.sub(r'```json\s*|\s*```', '', content).strip()
    try:
        parsed = json.loads(content_clean)
        return ensure_structure(parsed)
    except:
        pass
    
    # Try 3: Find JSON object
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return ensure_structure(parsed)
        except:
            pass
    
    return {"Tasks": [], "Deliverables": []}

def ensure_structure(parsed: Dict) -> Dict:
    """Ensure proper structure."""
    if "Tasks" not in parsed:
        parsed["Tasks"] = []
    if "Deliverables" not in parsed:
        parsed["Deliverables"] = []
    return parsed

# ---------- OPTIMIZED PROCESSING FUNCTION ----------
def process_file_dual_api(file_path: Path | io.BytesIO, file_extension: str, 
                          file_name: str, model: str) -> Dict:
    """
    OPTIMIZED VERSION v6.3:
    - Process 2 chunks at a time (one per API account)
    - Smart context carryover
    - Larger chunks (6 pages)
    - Batched logging
    - Staggered requests with backoff
    - Post-processing deduplication
    """
    
    start_time = time.time()
    
    logger.info(f"\n{'#'*100}")
    logger.info(f"# SARA v6.3 STRICT REGEX - START PROCESSING")
    logger.info(f"# File: {file_name}")
    logger.info(f"# Method: Paired Parallel with Backoff and Dedup")
    logger.info(f"# Timestamp: {datetime.now()}")
    logger.info(f"{'#'*100}\n")
    
    with st.spinner("📖 Reading document..."):
        full_text = clean_text(read_file(file_path, file_extension))
        if not full_text:
            logger.error("Failed to read file content")
            return {"Tasks": [], "Deliverables": []}
        logger.info(f"✅ Document read: {len(full_text)} chars")
    
    with st.spinner("✂️ Splitting document into chunks..."):
        chunks = split_into_page_chunks(full_text)
        if not chunks:
            logger.error("Failed to split document")
            return {"Tasks": [], "Deliverables": []}
        logger.info(f"✅ Document split into {len(chunks)} chunks")
    
    accounts = list(OPENROUTER_API_KEYS.keys())  # ['Account A', 'Account B']
    results = [None] * len(chunks)  # Preserve order
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear log queue
    log_queue.clear()
    
    # Process in pairs
    for i in range(0, len(chunks), 2):
        pair_start = time.time()
        
        # Get pair of chunks (or single if odd)
        chunk_pairs = []
        for j in range(2):
            idx = i + j
            if idx < len(chunks):
                chunk_pairs.append((idx, chunks[idx]))
        
        # Process pair in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            for pair_idx, (chunk_idx, chunk) in enumerate(chunk_pairs):
                api_account = accounts[pair_idx % 2]  # Alternate A/B
                api_key = OPENROUTER_API_KEYS[api_account]
                chunk_name = f"Chunk {chunk_idx+1}/{len(chunks)}"
                
                # Smart context from previous
                context = ""
                if chunk_idx > 0:
                    prev_result = results[chunk_idx - 1]
                    if prev_result:
                        context = extract_smart_context(prev_result, chunks[chunk_idx - 1])
                
                future = executor.submit(
                    extract_with_api,
                    chunk, chunk_name, api_account, api_key, model, context
                )
                futures.append((chunk_idx, future))
            
            # Collect results
            for chunk_idx, future in futures:
                results[chunk_idx] = future.result()
        
        # Stagger delay between pairs
        pair_elapsed = time.time() - pair_start
        if pair_elapsed < 2.0 and i + 2 < len(chunks):
            time.sleep(2.0 - pair_elapsed)
        
        # Update progress
        progress = min((i + 2) / len(chunks), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processed {min(i + 2, len(chunks))}/{len(chunks)} chunks")
    
    # Write batched logs
    write_logs_batch()
    
    # Combine results with deduplication
    task_dict = defaultdict(list)
    deliverable_dict = defaultdict(list)
    
    for result in results:
        for task in result.get("Tasks", []):
            key = (task.get("Task", ""), task.get("Parent Task", ""), task.get("Task Summary", ""))
            if key not in task_dict:
                task_dict[key] = task
        
        for deliv in result.get("Deliverables", []):
            key = (deliv.get("Deliverable", ""), deliv.get("Parent Task", ""), deliv.get("Description", ""))
            if key not in deliverable_dict:
                deliverable_dict[key] = deliv
    
    all_tasks = list(task_dict.values())
    all_deliverables = list(deliverable_dict.values())
    
    for i, task in enumerate(all_tasks):
        task["Position"] = i + 1
        task["Source File"] = file_name
    
    for i, deliv in enumerate(all_deliverables):
        deliv["Position"] = i + 1
        deliv["Source File"] = file_name
    
    elapsed = round(time.time() - start_time, 2)
    
    logger.info(f"\n{'#'*100}")
    logger.info(f"# PROCESSING COMPLETE")
    logger.info(f"# Total Time: {elapsed}s")
    logger.info(f"# Total Chunks: {len(chunks)}")
    logger.info(f"# TOTAL: {len(all_tasks)} tasks, {len(all_deliverables)} deliverables (after dedup)")
    logger.info(f"{'#'*100}\n")
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Processed in {elapsed}s | {len(chunks)} chunks | {len(all_tasks)} tasks | {len(all_deliverables)} deliverables (deduped)")
    
    return {"Tasks": all_tasks, "Deliverables": all_deliverables}

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="SARA v6.3 - Strict Regex + Dedup", layout="wide")

st.title(" SARA AUTOMATION FOR REQUIREMENT ANALYSIS")


with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🤖 AI Model")
    selected_model_label = st.selectbox("Choose model:", list(OPENROUTER_MODELS.keys()))
    selected_model = OPENROUTER_MODELS[selected_model_label]
    
    st.info(f"""
    **Using:** {selected_model_label}
    
    **Processing:**
    - Strict regex filter before AI
    - 6 pages/chunk
    - Dedup by name/parent/summary
    - Logs: `logs/sara_log_*.log`
    """)

uploaded_files = st.file_uploader("📁 Upload PDF/DOCX/TXT", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    all_tasks = []
    all_deliverables = []
    total_start = time.time()
    
    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
        st.divider()
        st.subheader(f"📄 File {file_idx}/{len(uploaded_files)}: {uploaded_file.name}")
        
        file_extension = f".{uploaded_file.name.split('.')[-1]}"
        temp_file_path = temp_dir / uploaded_file.name
        
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            extracted = process_file_dual_api(temp_file_path, file_extension, uploaded_file.name, selected_model)
            
            if extracted["Tasks"]:
                st.write(f"**📋 Extracted {len(extracted['Tasks'])} Tasks**")
                tasks_df = pd.DataFrame(extracted["Tasks"])
                st.dataframe(tasks_df, use_container_width=True, height=300)
                
                tasks_excel = io.BytesIO()
                tasks_df.to_excel(tasks_excel, index=False)
                tasks_excel.seek(0)
                st.download_button("⬇️ Download Tasks", tasks_excel, f"{uploaded_file.name.split('.')[0]}_tasks.xlsx", key=f"tasks_{file_idx}")
                all_tasks.extend(extracted["Tasks"])
            else:
                st.warning("⚠️ No tasks extracted - check logs for details")
            
            if extracted["Deliverables"]:
                st.write(f"**📦 Extracted {len(extracted['Deliverables'])} Deliverables**")
                deliv_df = pd.DataFrame(extracted["Deliverables"])
                st.dataframe(deliv_df, use_container_width=True, height=300)
                
                deliv_excel = io.BytesIO()
                deliv_df.to_excel(deliv_excel, index=False)
                deliv_excel.seek(0)
                st.download_button("⬇️ Download Deliverables", deliv_excel, f"{uploaded_file.name.split('.')[0]}_deliverables.xlsx", key=f"deliv_{file_idx}")
                all_deliverables.extend(extracted["Deliverables"])
            else:
                st.warning("⚠️ No deliverables extracted - check logs for details")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Processing error: {e}")
        finally:
            if temp_file_path.exists():
                os.remove(temp_file_path)
    
    if all_tasks or all_deliverables:
        st.divider()
        st.subheader("📊 Aggregated Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tasks", len(all_tasks))
        with col2:
            st.metric("Total Deliverables", len(all_deliverables))
        with col3:
            st.metric("Time", f"{round(time.time() - total_start, 1)}s")

else:
    st.info("👆 Upload documents - check logs/sara_log_*.log after processing")
