import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# Download NLTK Data
# ---------------------------------
nltk.download('stopwords')

# ---------------------------------
# Text Preprocessing
# ---------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ---------------------------------
# Skill Extraction
# ---------------------------------
SKILLS = [
    "python",
    "machine learning",
    "deep learning",
    "sql",
    "nlp",
    "scikitlearn",
    "aws",
    "docker",
    "tensorflow",
    "java",
    "excel",
    "power bi"
]

def extract_skills(text):

    text = text.lower()

    skills_found = []

    for skill in SKILLS:
        if skill in text:
            skills_found.append(skill)

    return list(set(skills_found))

# ---------------------------------
# Similarity Calculation
# ---------------------------------
def calculate_similarity(resume_text, job_text):

    documents = [resume_text, job_text]

    tfidf = TfidfVectorizer()

    matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )

    return round(float(score[0][0]) * 100, 2)

# ---------------------------------
# Skill Weighting
# ---------------------------------
weights = {
    "python": 5,
    "machine learning": 5,
    "sql": 4,
    "aws": 3,
    "docker": 3
}

def calculate_skill_score(resume_skills):

    score = 0

    for skill in resume_skills:
        if skill in weights:
            score += weights[skill]

    return score

# ---------------------------------
# Load Job Description
# ---------------------------------
with open("../data/job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

job_description_clean = clean_text(job_description)

print("\n===== JOB DESCRIPTION =====\n")
print(job_description_clean)

job_skills = extract_skills(job_description_clean)

print("\nJob Skills Found:")
print(job_skills)

# ---------------------------------
# Process Resumes
# ---------------------------------
results = []

resume_folder = "../data/resumes"

for resume_file in os.listdir(resume_folder):

    if resume_file.endswith(".txt"):

        path = os.path.join(
            resume_folder,
            resume_file
        )

        with open(path, "r", encoding="utf-8") as f:
            resume = f.read()

        resume_clean = clean_text(resume)

        resume_skills = extract_skills(
            resume_clean
        )

        similarity_score = calculate_similarity(
            resume_clean,
            job_description_clean
        )

        skill_score = calculate_skill_score(
            resume_skills
        )

        missing_skills = list(
            set(job_skills) -
            set(resume_skills)
        )

        print(
            f"{resume_file} -> Similarity: {similarity_score}"
        )

        results.append({
            "candidate": resume_file,
            "similarity_score": similarity_score,
            "skill_score": skill_score,
            "skills_found": resume_skills,
            "missing_skills": missing_skills
        })

# ---------------------------------
# Ranking
# ---------------------------------
ranking = pd.DataFrame(results)

ranking = ranking.sort_values(
    by=["similarity_score", "skill_score"],
    ascending=False
)

print("\n===== CANDIDATE RANKING =====\n")
print(ranking)

# ---------------------------------
# Save Results
# ---------------------------------
os.makedirs("../results", exist_ok=True)

ranking.to_csv(
    "../results/candidate_ranking.csv",
    index=False
)

print(
    "\nResults saved to results/candidate_ranking.csv"
)

# ---------------------------------
# Best Candidate
# ---------------------------------
top_candidate = ranking.iloc[0]

print("\n===== BEST CANDIDATE =====")
print("Candidate:", top_candidate["candidate"])
print("Similarity Score:", top_candidate["similarity_score"])
print("Skill Score:", top_candidate["skill_score"])
print("Skills Found:", top_candidate["skills_found"])
print("Missing Skills:", top_candidate["missing_skills"])

# ---------------------------------
# Visualization
# ---------------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    ranking["candidate"],
    ranking["similarity_score"],
    color="skyblue"
)

plt.xlabel("Candidates")
plt.ylabel("Similarity Score")
plt.title("Resume Screening Candidate Ranking")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()