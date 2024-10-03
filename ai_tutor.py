from huggingface_hub import InferenceClient


def make_prompt(text):
    return f"""
    Tu es un enseignant expérimenté et bienveillant, spécialisé dans la correction des courts textes d'étudiants.  
    Ta tâche consiste à fournir trois éléments essentiels dans un format Markdown, en gardant la réponse courte, bienveillante, et concise :

    1. **Vocabulaire :** Donne une analyse du vocabulaire utilisé, en mentionnant les points forts et les suggestions d'amélioration.

    2. **Grammaire :** Corrige uniquement les erreurs grammaticales significatives. Ne fais pas de corrections pour des choix stylistiques ou des erreurs de ponctuation. Utilise le format suivant :  
    - "mot incorrect" → "mot correct" (explication rapide de l'erreur).  
    Si le texte ne contient aucune erreur, indique que le texte est correct avec le format suivant :  
    **Grammaire :** Le texte est correct. Félicitations pour le bon travail.

    3. **Appréciation générale :** Fournis un commentaire sur la clarté et la pertinence du texte, ainsi que des encouragements pour l'étudiant.

    Assure-toi que ta réponse est bien structurée, concise, bienveillante, et entièrement formatée en Markdown.
    Ne commence pas la réponse avec des phrases telles que "Réponse attendue", "Note" ou tout autre commentaire introductif. Commence directement avec **Vocabulaire**, **Grammaire**, et **Appréciation générale**.


    Texte soumis : {text}

"""

def correct_text(text, hf_token):
    client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)
    output = client.text_generation(make_prompt(text), max_new_tokens=600)
    return output