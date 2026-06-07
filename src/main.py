import os
import matplotlib.pyplot as plt

from preprocess import clean_text
from skill_extractor import extract_skills
from similarity import calculate_similarity
from ranking import rank_candidates
from scoring import calculate_skill_score

# Load Job Description
with open("data/job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

job_description_clean = clean_text(job_description)

# Extract required skills from Job Description
job_skills = extract_skills(job_description_clean)

results = []

resume_folder = "data/resumes"

# Process all resumes
for file in os.listdir(resume_folder):

    if file.endswith(".txt"):

        path = os.path.join(resume_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            resume = f.read()

        # Clean resume text
        resume_clean = clean_text(resume)

        # Extract skills from resume
        resume_skills = extract_skills(resume_clean)

        # Calculate resume-job similarity
        similarity = calculate_similarity(
            resume_clean,
            job_description_clean
        )

        # Calculate weighted skill score
        skill_score = calculate_skill_score(
            resume_skills
        )

        # Find missing skills
        missing_skills = list(
            set(job_skills) - set(resume_skills)
        )

        # Store results
        results.append({
            "candidate": file,
            "similarity_score": similarity,
            "skill_score": skill_score,
            "skills_found": resume_skills,
            "missing_skills": missing_skills
        })

# Rank candidates
ranking = rank_candidates(results)

# Display rankings
print("\n===== Candidate Ranking =====\n")
print(ranking)

job_description_clean = clean_text(job_description)

print("Original Job Description:")
print(job_description)

print("\nCleaned Job Description:")
print(job_description_clean)

print("\nLength:", len(job_description_clean))

# Save results
ranking.to_csv(
    "results/candidate_ranking.csv",
    index=False
)

print("\nResults saved to results/candidate_ranking.csv")

# Visualization
plt.figure(figsize=(10, 5))

plt.bar(
    ranking["candidate"],
    ranking["similarity_score"],
    color="skyblue"
)

plt.xticks(rotation=45)
plt.ylabel("Similarity Score")
plt.xlabel("Candidates")
plt.title("Candidate Ranking Based on Resume Screening")
plt.tight_layout()

plt.show()