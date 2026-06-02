import os
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

# GEMINI_API_KEY removed - add your own key to .env if needed
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

try:
    response = model.generate_content(
        "Reply with only: OK"
    )

    print("Connected:")
    print(response.text)

except Exception as e:
    print("Error:")
    print(e)


def classify_abstract(
    abstract: str
):

    prompt = f"""
You are a binary classifier for research paper abstracts.

Classify a paper as Relevant if it focuses on inflation analysis, inflation forecasting, inflation prediction, CPI forecasting, price index modelling, monetary policy and inflation dynamics, or the application of machine learning, artificial intelligence, deep learning, data mining, or advanced predictive analytics to inflation-related problems.

Relevant topics include:

* Inflation forecasting or prediction
* CPI, WPI, PPI, or price index forecasting
* Inflation modelling and analysis
* Monetary policy and inflation dynamics
* Inflation determinants and drivers
* Machine learning for inflation prediction
* AI, deep learning, ensemble learning, NLP, or data-driven approaches applied to inflation research
* Economic forecasting where inflation is a primary target variable

Classify as Not Relevant if the paper:

* Does not focus on inflation or price-level analysis
* Focuses on unrelated economic indicators without substantial inflation analysis
* Discusses machine learning without any inflation-related application
* Concerns general economics, finance, or forecasting where inflation is not a central topic

Return ONLY one of the following:

Relevant

or

Not Relevant


Abstract:
{abstract}
"""

    response = model.generate_content(
        prompt
    )

    answer = (
        response.text
        .strip()
        .lower()
    )

    if "not relevant" in answer:

        return {
            "prediction": 0,
            "label": "Not Relevant"
        }

    return {
        "prediction": 1,
        "label": "Relevant"
    }