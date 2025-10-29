import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ==========================
# Synthetic Data Generation
# ==========================
np.random.seed(42)
n = 1000

education = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.2, 0.4, 0.22, 0.03], size=n)
supervisory = np.random.exponential(scale=2, size=n).clip(0, 20)
private_sector = np.random.choice([0, 1], p=[0.6, 0.4], size=n)
federal_years = np.random.randint(0, 31, size=n)
tech_certs = np.random.choice([0, 1, 2, 3, 4, 5],
                              p=[0.4, 0.35, 0.15, 0.06, 0.03, 0.01],
                              size=n)

success_logit = (
    0.4 * education +
    0.3 * supervisory +
    0.5 * private_sector +
    0.25 * np.exp(-((federal_years - 10) ** 2) / 80) +
    0.3 * np.minimum(tech_certs, 2) -
    2
)
success_prob = 1 / (1 + np.exp(-success_logit))
success = np.random.binomial(1, success_prob)

df = pd.DataFrame({
    "education": education,
    "supervisory": supervisory,
    "private_sector": private_sector,
    "federal_years": federal_years,
    "tech_certs": tech_certs,
    "success": success
})

X = df[["education", "supervisory", "private_sector", "federal_years", "tech_certs"]]
y = df["success"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ==========================
# Streamlit Interface
# ==========================
st.set_page_config(page_title="Federal Career Pivot Buddy", layout="centered")
st.markdown(
    "<style>.block-container{max-width:700px;padding-top:2rem;text-align:center;}</style>",
    unsafe_allow_html=True
)

st.title("🌿 Federal Career Pivot Buddy")

st.write(
    '''
Welcome to **Federal Career Pivot Buddy**, a tool to help you explore your readiness for a **successful career pivot** away from federal service.

For this tool, *success* is defined as obtaining employment that is effective, satisfying, and sustainable in an environment where your background, skills, and motivations are utilized and celebrated *within a few months after separation.*

Federal Career Pivot Buddy offers **insights** inspired by *strength-based social work practice theory*, nurturing your potential while highlighting realistic areas for growth.
'''
)

st.markdown("---")
st.subheader("🔧 Your Profile")

education_map = {
    "High School or GED": 0,
    "Associate's or Trade School": 1,
    "Bachelor’s Degree": 2,
    "Master’s Degree": 3,
    "Doctorate": 4
}
education_choice = st.selectbox("Highest Education Level", list(education_map.keys()))
education_level = education_map[education_choice]

supervisory_years = st.slider("Years of Supervisory Experience", 0, 20, 0)
private_sector_choice = st.selectbox("5+ Years Private Sector Work Experience", ["No", "Yes"])
private_sector_val = 1 if private_sector_choice == "Yes" else 0
federal_years = st.slider("Years of Federal Service", 0, 30, 5)
tech_certs = st.slider("Number of Technical Certifications", 0, 5, 1)

input_data = np.array([[education_level, supervisory_years, private_sector_val, federal_years, tech_certs]])
pred_prob = model.predict_proba(input_data)[0][1]
prob_percent = round(pred_prob * 100, 1)

# Growth-themed labels
def readiness_label(score):
    if score < 40:
        return "🌱 Growing Roots — building a strong foundation for your next step."
    elif score < 55:
        return "🌿 Emerging Potential — developing strengths that are beginning to bloom."
    elif score < 70:
        return "🌸 Blossoming Talent — your readiness is flourishing with solid potential."
    elif score < 85:
        return "🌼 Thriving Momentum — your experience and skills are well-aligned for a pivot."
    else:
        return "🌻 Ready to Shine — your preparation and growth make you poised for success!"

readiness_text = readiness_label(prob_percent)

st.markdown("---")
st.subheader("📈 Your Readiness Level")
st.markdown(f"### {readiness_text}")
st.caption(f"(Estimated probability of successful pivot: **{prob_percent}%**)")
st.markdown("---")

strengths = []
suggestions = []

if education_level >= 3:
    strengths.append("Your advanced education enhances credibility and adaptability.")
elif education_level == 2:
    strengths.append("Your bachelor's degree provides a solid professional foundation.")
else:
    suggestions.append("Consider pursuing continuing education or certifications to expand opportunities.")

if supervisory_years > 10:
    strengths.append("Extensive supervisory experience demonstrates strong leadership.")
elif supervisory_years > 0:
    strengths.append("Your supervisory background highlights emerging leadership skills.")
else:
    suggestions.append("Seek leadership opportunities through mentoring, volunteering, or leading projects.")

if private_sector_val == 1:
    strengths.append("Private sector experience shows agility and understanding of diverse workplaces.")
else:
    suggestions.append("Emphasize collaboration and cross-functional results to showcase adaptability.")

if 3 <= federal_years <= 15:
    strengths.append("Your federal tenure reflects balanced domain knowledge and openness to growth.")
elif federal_years < 3:
    suggestions.append("Building additional federal experience can strengthen your credibility and perspective.")
else:
    suggestions.append("Emphasize innovation and continuous learning to highlight versatility.")

if tech_certs >= 2:
    strengths.append("Your technical certifications show initiative and digital fluency.")
elif tech_certs == 1:
    strengths.append("Your certification reflects professional growth; consider expanding your tech toolkit.")
else:
    suggestions.append("Consider earning certifications in areas like data, AI, or project management.")

st.markdown("### 💪 Strengths")
for s in strengths:
    st.markdown(f"- {s}")

st.markdown("### 🧭 Suggestions for Growth")
for s in suggestions:
    st.markdown(f"- {s}")

st.markdown("---")
st.caption("This tool uses synthetic data for educational demonstration. Designed with ❤️ for federal professionals exploring their next chapter.")
