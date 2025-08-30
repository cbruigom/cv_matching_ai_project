# CV Matcher with Embeddings + GPT-4

This project matches CVs (PDFs) against job title and description (CSV) with OpenAI LLM's using a **hybrid approach**:

1. **Embeddings (OpenAI `text-embedding-3-small`)**  
   - Computes semantic similarity between CVs and job descriptions.  
   - Cheap, fast, scalable filter.  

2. **GPT-4 structured evaluation**  
   - Evaluates CVs against job descriptions using **subscores**:  
     - Direct alignment (0–10)  
     - Transferable skills (0–10)  
     - Education/training (0–10)  
   - Produces a **composite GPT score** (default: 40/30/30 weighting).  
   - Provides recruiter-style reasoning for each CV.  

3. **Combined scoring**  
   - Blends embeddings + GPT scores into a final ranking.  
   - Default weighting: **60% GPT composite / 40% embeddings**.  
   - Balances raw semantic closeness with contextual recruiter judgment.  

---

## 🚀 How It Works

1. **Load job postings** from `jobs/job_postings.csv` (must contain `job_title` and `description` columns).  
2. **Load CVs** from the `cvs/` folder (PDF format).  
3. **Generate embeddings** for CVs and job descriptions (cached in `embeddings_cache.pkl`).  
4. **Rank by embeddings** using cosine similarity (0–1 scale).  
5. **Send CVs to GPT-4** for structured scoring.  
6. **Compute combined score** and print all three rankings:
   - Embeddings ranking  
   - GPT-4 ranking with subscores + reasoning  
   - Combined ranking  

---

##   Environment Variables
OPENAI_API_KEY=your_openai_api_key_here

---

## ⚙⚙️ Configuration Levers

These levers are set at the top of `cvmatcher.py`:

- **Embeddings Top-N (printing only)**  
TOP_N = 10

- **GPT Top-N (CVs evaluated by GPT)**
GPT_TOP_N = None (None → all CVs sent to GPT, 3 → only top 3 from embeddings sent)

- **GPT Subscore Weighting**
composite = (0.4 * alignment) + (0.3 * transferable) + (0.3 * education) - Adjust to shift GPT emphasis between direct alignment, transferable skills, and education.

- Combined Weighting Score
combined_score = (0.6 * gpt_score) + (0.4 * emb_score) - Adjust GPT vs embeddings influence on the final ranking.

---

##  🗂  Project Structure

```text
.
├── cvs/                  # Folder for CV PDFs (not committed to git)
│   └── sample_cv.pdf     # An anonymized example
├── jobs/
│   └── job_postings.csv  # Job postings CSV
├── embeddings_cache.pkl  # Pickled embeddings cache (ignored by git)
├── cvmatcher.py          # Main script
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
├── LICENSE               # License file (MIT by default)
└── README.md             # This file
```
---

🔑 Requirements

Python 3.9+

OpenAI API key (set as environment variable: OPENAI_API_KEY)

Install dependencies:

pip install -r requirements.txt

___

▶️ Usage

Place CV PDFs in cvs/

Add job types to jobs/job_postings.csv with columns:

job_title

description

Run:

python cvmatcher.py

You’ll see three outputs per job:

Embedding-based ranking (semantic closeness only)

GPT-4 ranking with subscores (alignment, transferable, education) + composite

Combined ranking (60% GPT / 40% embeddings by default)

---

📊 Example Output
Job: IT Data Analyst

Embedding-based Ranking (semantic similarity):
    Rank 1: CandidateA.pdf (0.262)
    Rank 2: CandidateB.pdf (0.230)

GPT-4 Scoring (subscores and composite):
    CandidateA.pdf
      Alignment: 8/10, Transferable: 6/10, Education: 7/10
      Composite Score: 7.2/10
      Reason: Strong IT background with relevant project work.

Combined Ranking (60% GPT / 40% Embeddings):
    Rank 1: CandidateA.pdf (0.751)
    Rank 2: CandidateB.pdf (0.642)

---

🔒 Privacy Considerations

OpenAI API does not train on your data.

CVs may contain personal information → best practice:

Strip names, emails, phone numbers before sending to API.

Keep only education, skills, and experience.

Add CVs to .gitignore to avoid committing private data to GitHub.

---

🛠 Future Improvements

Export rankings to CSV/JSON.

Chunk large CVs to handle GPT token limits.

Visualize similarity scores with charts.

Add anonymization step before API calls.


License
The contents of this repository are licensed under the MIT License. Feel free to use, modify, and distribute the code and accompanying files as needed.
