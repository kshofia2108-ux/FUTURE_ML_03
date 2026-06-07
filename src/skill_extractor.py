import spacy

nlp = spacy.load("en_core_web_sm")

SKILLS = [
    "python",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "sql",
    "java",
    "data analysis",
    "nlp",
    "excel",
    "power bi",
    "tableau",
    "scikit-learn",
    "aws",
    "docker",
    "git"
]

def extract_skills(text):

    text = text.lower()

    extracted = []

    for skill in SKILLS:
        if skill.lower() in text:
            extracted.append(skill)

    return list(set(extracted))