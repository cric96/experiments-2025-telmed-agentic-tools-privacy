from langchain_google_genai import ChatGoogleGenerativeAI
from patients_generation import multiple_patients_generation
from attacks_generation import generate_attacks_for_patients
import json

chat = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-2.5-flash")

# Generate patients
result = multiple_patients_generation(total=10, how_many=2, llm=chat)

# Store patients as json
with open("generated_patients.json", "w") as f:
    json.dump([patient.model_dump() for patient in result], f, indent=2)

# Generate attack questions for each patient
attacks = generate_attacks_for_patients(result, llm=chat)

# Store attacks as json
with open("generated_attacks.json", "w") as f:
    json.dump(attacks, f, indent=2)
