import streamlit as st
import pandas as pd
import base64
from transformers import pipeline
from ai_tutor import correct_text

correction_text = """
**Vocabulaire :**
Le vocabulaire utilisé est simple mais efficace. Le choix des mots est approprié pour décrire les qualités de Beyoncé. Cependant, il y a quelques mots qui pourraient être remplacés pour ajouter de la variété. Par exemple, au lieu de "très bien", vous pourriez utiliser "exceptionnellement" ou "avec talent". De plus, "incroyable" est un adjectif qui est souvent utilisé, vous pourriez essayer "époustouflant" ou "mémorable" pour ajouter de la diversité.

**Grammaire :**
- "Ma star préférée est Beyoncé." → "Ma star préférée est Beyoncé." (aucune erreur)
- "Elle chante très bien et ses concerts sont incroyable." → "Elle chante très bien et ses concerts sont incroyables." (accord du pluriel)
- "J'aime qu'elle parle de sujets importants comme l'égalité." → "J'aime qu'elle parle de sujets importants, comme l'égalité." (virgule de séparation)
- "Mais je trouve que certaines de ses chansons sont trop répétitifs." → "Mais je trouve que certaines de ses chansons sont trop répétitives." (accord du féminin)

**Appréciation générale :**
Votre texte est clair et facile à comprendre. Vous avez réussi à exprimer vos opinions sur Beyoncé de manière concise. N'oubliez pas de varier votre vocabulaire et de vérifier l'accord grammatical pour améliorer votre écriture. Continuez à écrire et à vous exprimer avec confiance!

"""


# read csv and get questions by level
df = pd.read_csv('content/question_demo_bsf.csv')
beginner = df[df['level'] == 'beginner']['question'].tolist()
intermediate = df[df['level'] == 'intermediate']['question'].tolist()
advanced = df[df['level'] == 'advanced']['question'].tolist()

# model to predict the level of the input text
classifier = pipeline("sentiment-analysis", model="aapoliakova/cls_level_bsf")

text = beginner[0]

# put image and buttin in the center
st.markdown("<style>div.stButton {display: flex; justify-content: center;}</style>", unsafe_allow_html=True)
st.markdown("<style>div.stImage {display: flex; justify-content: center;}</style>", unsafe_allow_html=True)

logo_path = "content/Logo_Karibu.png"
icon_path = "content/icon_ecrit.svg"



# logo
st.image(logo_path)

# block with icon and exercice type
file_ = open(icon_path, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.html(f"""
    <div style="background: linear-gradient(90deg, #E23337, #E41AAC); padding: 6px; border-radius: 8px; display: flex; align-items: left;">
        <img src="data:image/svg+xml;base64,{data_url}">
        <p style="color: white; margin: 8px;">EXPRESSION ÉCRITE</p>
    </div>
""")


# Exercice description
st.write("\nÀ vous de jouer ! " + text)

# Input field
text_input = st.text_area("Écrivez ici, au moins 30 mots...", height=200)

# Button
if st.button('Valider', type="primary"):
    if len(text_input) < 30:
        st.warning("Le texte doit contenir au moins 30 mots.")
    else:
        st.subheader("Correction")
        with st.spinner('Wait for it...'):
    # time.sleep(5)
        # get correction from the model
            st.write(correct_text(text_input))

        # get the level of the input text
        res = classifier(text_input)
        st.write('level of your text is ', res[0]['label'])

    # st.html("""
    #     <div style="background-color: #CACACA; padding: 8px; border-radius: 8px; display: flex; align-items: left;">
    #         <p>your text</p>
    #     </div>
    # """)
    # print(classifier(text_input))
