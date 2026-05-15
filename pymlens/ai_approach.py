from groq import Groq
def generate_dna_report(data,api_key,problem_type,experiment_name):
    client=Groq(api_key=api_key)
    prompt = f"""
You are an ML expert. Write a concise DNA Report for each model below.
Be beginner-friendly but specific. Reference actual numbers only.

Experiment: {experiment_name}
Problem Type: {problem_type}
Data: {data}

For EACH model write exactly this format, nothing extra:

[Model Name]
- Accuracy: <what this score means in 1 line>
- F1/Precision/Recall: <what these tell us in 1 line>
- CV Score: <stable or unstable and why>
- Overfitting: <compare train vs val, 1 line>
- Fix: <one specific improvement>

---

VERDICT: <best model name> — <reason in 1 line>

Rules:
- No intros or summaries
- No generic advice
- Max 6 lines per model
"""
    response=client.chat.completions.create(model="llama-3.1-8b-instant",messages=[{"role":"user","content":prompt}])
    return response.choices[0].message.content

