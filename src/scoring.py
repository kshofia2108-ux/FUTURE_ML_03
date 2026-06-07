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