import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ==========================
# Synthetic Data Generation
# ==========================
np.random.seed(42)
n = 1000

# Education levels (0=HS, 1=Assoc/Trade, 2=Bachelor, 3=Master, 4=PhD)
education = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.2, 0.4, 0.22, 0.03], size=n)

# Supervisory experience (0–20, skewed toward 0)
supervisory = np.random.exponential(scale=2, size=n).clip(0, 20)

# Binary: 5+ years private sector experience (Yes=1, No=0)
private_sector = np.random.choice([0, 1], p=[0.6, 0.4], size=n)

# Federal service years (0–30)
federal_years = np.random.randint(0, 31, size=n)

# Technical certifications (0–5, 0–1 most common)
tech_certs = np.random.choice([0, 1, 2, 3, 4, 5],
                              p=[0.4, 0.35, 0.15, 0.06, 0.03, 0.01],
                              size=n)

# --- Define true relationship for success probability ---
success_logit = (
    0.4 * education +
    0.3 * supervisory +
    0.5 * private_sector +  # binary boost
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

# ==========================
# Train Model
# ==========================
X = df[["education", "supervisory", "private_sector", "federal_years", "tech_certs"]]
y = df["success"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ==========================
# Streamlit Interface
# ==========================
st.set_page_config(page_title="Federal Career Pivot Buddy", layout="centered")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 750px;
        padding-top: 2rem;
        text-align: center;
    }
    .stSlider label, .stSelectbox label {
        font-size: 1.2rem !important;
    }
    .stMarkdown {
        font-size: 1.1rem !important;
        text-align: left !important;
    }
    .st-bb {
        text-align: left !important;
    }
    .st-bb p {
        text-align: left !important;
    }
    .stSlider > div {
        width: 90% !important;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Federal Career Pivot Buddy 👋")

st.write("""
Welcome to **Federal Career Pivot Buddy**, a tool to help you explore your readiness for a **successful career pivot** away from federal service.

For this tool, *success* is defined as obtaining employment that is effective, satisfying, and sustainable in an environment where your background, skills, and motivations are utilized and celebrated *within a few months after separation*.  

Long job hunts can be financially, mentally, and physically straining, so this tool emphasizes readiness for the *ideal outcome*: a timely and fulfilling transition.

Federal Career Pivot Buddy provides **insights** inspired by *strengths-based social work practice theory*, highlighting the assets you already bring to the table while suggesting realistic areas for continued growth.  
Remember, a computer program cannot capture the richness of your professional journey — these insights are intended for reflection, not as individualized career advice.
""")

# ==========================
# User Inputs
# ==========================
st.markdown("---")
st.subheader("🔧 Your Profile")

# Education dropdown
education_map = {
    "High School or GED": 0,
    "Associate’s or Trade School": 1,
    "Bachelor’s Degree": 2,
    "Master’s Degree": 3,
    "Doctorate": 4
}
education_choice = st.selectbox("Highest Education Level", list(education_map.keys()))
education_level = education_map[education_choice]

# Supervisory experience
supervisory_years = st.slider("Years of Supervisory Experience", 0, 20, 0)

# Private sector binary dropdown
private_sector_choice = st.selectbox("5+ Years Private Sector Work Experience", ["No", "Yes"])
private_sector_val = 1 if private_sector_choice == "Yes" else 0

# Federal service years
federal_years = st.slider("Years of Federal Service", 0, 30, 5)

# Technical certifications
tech_certs = st.slider("Number of Technical Certifications or Other Measures of Digital Skill Mastery", 0, 5, 1)

# ==========================
# Prediction
# ==========================
input_data = np.array([[education_level, supervisory_years, private_sector_val, federal_years, tech_certs]])
pred_prob = model.predict_proba(input_data)[0][1]
prob_percent = round(pred_prob * 100, 1)

# ==========================
# Growth-Themed Readiness Mapping
# ==========================
if prob_percent < 40:
    readiness_level = "Planting new seeds 🌱"
    level_num = 1
    reflection = "You’re at the beginning of an exciting growth journey — a time to explore new directions and invest in the roots of your next opportunity."
elif prob_percent < 60:
    readiness_level = "Taking root 🌿"
    level_num = 2
    reflection = "You’re grounding yourself — building skills, experience, and clarity that will soon support meaningful professional growth."
elif prob_percent < 75:
    readiness_level = "Branching out 🌳"
    level_num = 3
    reflection = "You’re expanding your reach and applying your strengths — a strong stage for networking, refining goals, and exploring options."
elif prob_percent < 90:
    readiness_level = "In full bloom 🌸"
    level_num = 4
    reflection = "Your strengths are shining. You’re well-prepared for your next move and showing readiness to adapt and thrive in new environments."
else:
    readiness_level = "Thriving in new soil 🌺"
    level_num = 5
    reflection = "You’re ready to flourish — your skills, experience, and mindset align beautifully for a confident, fulfilling career transition."

# ==========================
# Display Readiness
# ==========================
st.markdown("---")
st.subheader("Your Career Pivot Readiness")
st.markdown(f"### **Level {level_num}: {readiness_level}**")
st.markdown(f"_{reflection}_")
st.caption(f"(Model-estimated readiness: {prob_percent}% — interpreted through a strengths-based, growth-focused lens.)")
st.markdown("---")

# ==========================
# Strengths & Suggestions
# ==========================
strengths = []
suggestions = []

# Education
if education_level >= 3:
    strengths.append("Your advanced education is a major strength that enhances credibility and adaptability.")
elif education_level == 2:
    strengths.append("Your bachelor's degree provides a solid foundation for professional growth.")
else:
    suggestions.append("Consider additional training, certificates, or continuing education to expand opportunities.")

# Supervisory Experience
if supervisory_years > 10:
    strengths.append("Extensive supervisory experience shows advanced leadership and decision-making abilities.")
elif supervisory_years > 0:
    strengths.append("Your supervisory background highlights emerging leadership skills.")
else:
    suggestions.append("Explore leadership opportunities through mentoring, project coordination, or volunteering inside or outside the office.")

# Private Sector Experience
if private_sector_val == 1:
    strengths.append("Private sector experience adds agility and an understanding of the industry landscape.")
else:
    suggestions.append("Highlight collaboration with external organizations and results-focused achievements to show adaptability.")

# Federal Years
if 3 <= federal_years <= 15:
    strengths.append("Your federal tenure demonstrates both domain knowledge and flexibility for new challenges.")
elif federal_years < 3:
    suggestions.append("Building additional federal experience could strengthen your credibility and confidence in pivots.")
else:
    suggestions.append("Emphasize continuous learning and innovation to counter perceived long-tenure specialization.")

# Technical Certifications
if tech_certs >= 2:
    strengths.append("Your technical certifications demonstrate initiative and relevance in today’s digital landscape.")
elif tech_certs == 1:
    strengths.append("Your certification shows a commitment to professional growth. Consider pursuing additional certifications in data, analytics, or technology to expand your options.")
else:
    suggestions.append("Consider pursuing certifications in data, analytics, or technology to expand your options.")

# ==========================
# Display Insights
# ==========================
st.markdown("### 💪 Strengths")
for s in strengths:
    st.markdown(f"- {s}")

st.markdown("### 🧭 Suggestions for Growth")
for s in suggestions:
    st.markdown(f"- {s}")

st.markdown("---")
st.caption(
    "This tool uses synthetic data to simulate realistic patterns for demonstration and educational purposes. "
    "It was developed using Streamlit, NumPy, and Logistic Regression in a low-code, AI-assisted Python environment optimized for Jupyter. "
    "Designed for feds, by feds :)"
)
