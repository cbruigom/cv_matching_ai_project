import os
import pickle
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import json

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CV_FOLDER = "cvs"                    # Folder containing PDF CVs
JOB_CSV = "jobs/job_postings.csv"    # CSV with job postings
EMBEDDINGS_FILE = "embeddings_cache.pkl"
TOP_N = 10                           # How many CVs to show for embeddings
GPT_TOP_N = None                     # None = send all CVs to GPT for ranking

# --- INIT OPENAI CLIENT ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- HELPER FUNCTIONS ---

def get_embedding(text):
    """Get embedding vector from OpenAI API for given text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def get_gpt_ranking(job_title, job_description, cv_texts):
    """
    Call GPT-4 to rank CVs using three subscores:
      1. Direct alignment (0–10)
      2. Transferable skills (0–10)
      3. Education/training (0–10)
    GPT must return JSON with fields:
      - cv_name
      - alignment_score
      - transferable_score
      - education_score
      - reason
    """
    prompt = f"""
You are a recruiter. Evaluate the following CVs for the job: {job_title}.
Job description:
{job_description}

For each CV, provide scores (0–10) for:
1. Direct alignment with the role requirements (skills, technical expertise, relevant experience).  
2. Transferable skills (research, data analysis, problem-solving, leadership, adaptability, communication, etc.).  
3. Education and training (degrees, certifications, courses, professional development).  

Then provide a short reason for your evaluation.

Return ONLY a JSON array of objects with this structure:
[
  {{
    "cv_name": "filename.pdf",
    "alignment_score": int,
    "transferable_score": int,
    "education_score": int,
    "reason": "short text explanation"
  }},
  ...
]
CVs:
"""
    for name, text in cv_texts.items():
        prompt += f"\n{name}:\n{text}\n"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        content = response.choices[0].message.content
        results = json.loads(content)   # Parse GPT JSON output
        return results
    except Exception as e:
        print("Error parsing GPT output:", e)
        print("Raw GPT output:", response.choices[0].message.content)
        return []


# --- LOAD JOBS CSV ---
if os.path.exists(JOB_CSV):
    jobs_df = pd.read_csv(JOB_CSV)
else:
    raise FileNotFoundError(f"{JOB_CSV} not found. Create it with columns: job_title,description")

# --- LOAD PDF CVs ---
cv_texts = {}
for file_name in os.listdir(CV_FOLDER):
    if file_name.endswith(".pdf"):
        path = os.path.join(CV_FOLDER, file_name)
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        cv_texts[file_name] = text

if not cv_texts:
    raise ValueError(f"No PDF CVs found in {CV_FOLDER}/ folder.")

# --- LOAD OR CREATE EMBEDDINGS CACHE ---
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_cache = pickle.load(f)
else:
    embeddings_cache = {}

# --- Compute embeddings for jobs ---
for idx, row in jobs_df.iterrows():
    job_key = f"job_{idx}"
    if job_key not in embeddings_cache:
        embeddings_cache[job_key] = get_embedding(row['description'])

# --- Compute embeddings for CVs ---
for cv_name, text in cv_texts.items():
    if cv_name not in embeddings_cache:
        embeddings_cache[cv_name] = get_embedding(text)

# --- Save embeddings cache ---
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(embeddings_cache, f)

# --- RANK CVs FOR EACH JOB ---
for idx, row in jobs_df.iterrows():
    job_key = f"job_{idx}"
    job_vector = embeddings_cache[job_key]

    # 1) --- Embedding similarity ranking ---
    similarities = []
    for cv_name, cv_vector in {k:v for k,v in embeddings_cache.items() if k in cv_texts}.items():
        score = cosine_similarity([job_vector], [cv_vector])[0][0]
        similarities.append((cv_name, score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\nJob: {row['job_title']}")

    print("  Embedding-based Ranking (semantic similarity):")
    for rank, (cv_name, score) in enumerate(similarities[:TOP_N], 1):
        print(f"    Rank {rank}: {cv_name} (similarity score: {score:.3f})")

    # 2) --- GPT scoring (alignment, transferable, education) ---
    top_cvs = cv_texts if GPT_TOP_N is None else {cv: cv_texts[cv] for cv, _ in similarities[:GPT_TOP_N]}
    gpt_results = get_gpt_ranking(row['job_title'], row['description'], top_cvs)

    if gpt_results:
        print("\n  GPT-4 Scoring (subscores and composite):")
        gpt_scores = {}
        for item in gpt_results:
            alignment = item["alignment_score"]
            transferable = item["transferable_score"]
            education = item["education_score"]

            # Composite GPT score (0–10)
            composite = (0.4 * alignment) + (0.3 * transferable) + (0.3 * education)
            gpt_scores[item["cv_name"]] = composite / 10  # normalize to 0–1

            print(f"    {item['cv_name']}:")
            print(f"      Alignment: {alignment}/10, Transferable: {transferable}/10, Education: {education}/10")
            print(f"      Composite Score: {composite:.2f}/10")
            print(f"      Reason: {item['reason']}")

        # 3) --- Combined score (60% GPT composite, 40% embeddings) ---
        combined = []
        for cv_name, emb_value in similarities:
            emb_score = emb_value  # already 0–1 scale
            gpt_score = gpt_scores.get(cv_name, 0)  # normalized 0–1
            combined_score = (0.6 * gpt_score) + (0.4 * emb_score)
            combined.append((cv_name, combined_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        print("\n  Combined Ranking (60% GPT / 40% Embeddings):")
        for rank, (cv_name, score) in enumerate(combined, 1):
            print(f"    Rank {rank}: {cv_name} (weighted score: {score:.3f})")

