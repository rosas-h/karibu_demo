from huggingface_hub import InferenceClient
import json


def make_prompt(text):
    template = """
    Tu es un enseignant expérimenté et bienveillant, spécialisé dans la correction des textes d'étudiants.
    Ta tâche consiste à fournir trois éléments essentiels dans un format Markdown, en gardant la réponse courte, bienveillante, et concise, uniquement comme ceci :

    1. **Vocabulaire :** Donne une analyse du vocabulaire utilisé, en mentionnant les points forts et les suggestions d'amélioration.

    2. **Grammaire :** Corrige toutes les erreurs de francais. Utilise le format suivant :
    - "mot incorrect" → "mot correct" (explication rapide de l'erreur).
    Si le texte ne contient aucune erreur, indique que le texte est correct avec le format suivant :
    **Grammaire :** Le texte est correct. Félicitations !

    3. **Appréciation générale :** Fournis un commentaire sur la clarté et la pertinence du texte, ainsi que des encouragements pour l'étudiant.

    Assure-toi que ta réponse est bien structurée, concise, bienveillante, et entièrement formatée en Markdown.
    Commence directement avec **Vocabulaire**, **Grammaire**, et **Appréciation générale**.
    Donne moi uniquement la réponse comme output

    Texte soumis : 
    {text}
    """
    return template

def correct_text(text, hf_token):
    debug_info = []
    debug_info.append(f"Debug: Input text: {repr(text)}")
    
    client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)
    
    prompt_template = make_prompt("{text}")
    debug_info.append(f"Debug: Prompt template: {repr(prompt_template)}")
    
    full_prompt = prompt_template.format(text=text)
    debug_info.append(f"Debug: Full prompt: {repr(full_prompt)}")
    
    try:
        output = client.text_generation(full_prompt, max_new_tokens=4000)
        debug_info.append(f"Debug: Raw output: {repr(output)}")
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        debug_info.append(f"Debug: Exception occurred: {error_msg}")
        return error_msg, debug_info
    
    return output, debug_info

def test_inference_client(hf_token):
    client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)
    test_prompt = "Translate the following to English: Bonjour, comment allez-vous?"
    try:
        output = client.text_generation(test_prompt, max_new_tokens=100)
        return f"Test output: {output}"
    except Exception as e:
        return f"Test exception: {str(e)}"
