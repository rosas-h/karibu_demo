import streamlit as st
import pandas as pd
import base64
import random
from transformers import pipeline
from ai_tutor import correct_text

token = st.secrets["hf_token"]

# read csv and get questions by level
df = pd.read_csv('content/question_demo_bsf.csv')
beginner = df[df['level'] == 'beginner']['question'].tolist()
intermediate = df[df['level'] == 'intermediate']['question'].tolist()
advanced = df[df['level'] == 'advanced']['question'].tolist()

# model to predict the level of the input text
classifier = pipeline("sentiment-analysis", model="aapoliakova/cls_level_bsf")

# initialize the session state
if 'count' not in st.session_state:
    st.session_state.count = 0

if 'text' not in st.session_state:
    st.session_state.text = random.choice(beginner)

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""


# put image and buttin in the center
st.markdown("<style>div.stButton {display: flex; justify-content: center;}</style>", unsafe_allow_html=True)
st.markdown("<style>div.stImage {display: flex; justify-content: center;}</style>", unsafe_allow_html=True)

logo_path = "content/Logo_Karibu.png"
logo_pleias = "content/logo_pleias.png"
icon_path = "content/icon_ecrit.svg"

def green_header(text):
    st.markdown(
        f"""
        <h2 style='color: #4ABC96;'>{text}</h1>
        """, unsafe_allow_html=True
    )


# logo
st.image(logo_path)

# block with icon and exercice type
def get_image(icon_path):
    file_ = open(icon_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url
# file_ = open(icon_path, "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()

st.html(f"""
    <div style="background: linear-gradient(90deg, #E23337, #E41AAC); padding: 6px; border-radius: 8px; display: flex; align-items: left;">
        <img src="data:image/svg+xml;base64,{get_image(icon_path)}">
        <p style="color: white; margin: 8px;">EXPRESSION ÉCRITE</p>
    </div>
""")


# Exercice description
st.write("\n")
st.write("À vous de jouer ! " + st.session_state.text)

# Input field
text_input = st.text_area("Ecrivez un texte d'au moins 30 mots", label_visibility="collapsed", height=200, placeholder="Ecrivez un texte d'au moins 30 mots", value=st.session_state.user_input)

# Button
if st.button('Valider', type="primary"):
    if len(text_input) < 30:
        st.warning("Le texte doit contenir au moins 30 mots.")
    else:
        # st.subheader('**:green[Correction]**')
        green_header('Correction')
        
        with st.spinner('Correction en cours, veuillez patienter...'):
        # get correction from the model
            st.write(correct_text(text_input, token))

        # Classifier and recomendation
        green_header('Recomendation')

        # predict the level of the input text
        res = classifier(text_input)[0]['label']

        # get random question based on the level
        if res == 'advanced':
            st.session_state.text = random.choice(advanced)
        elif res == 'intermediate':
            st.session_state.text = random.choice(intermediate)
        else:
            st.session_state.text = random.choice(beginner)
        st.write("Voici une nouvelle activité spécialement choisie pour vous, afin de corriger vos erreurs et perfectionner votre français.")
        if st.button("Commencer l'exercice suivant", type="primary"):
            st.session_state.count += 1
            st.session_state.user_input = ""
            st.experimental_rerun()
st.session_state.user_input = text_input

# st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
# st.image(logo_pleias, width=150)

# st.markdown("""
#     <div style='margin-top: 40px;'>
#         <img src='image.png' style='width: 150px; height: auto;'/>
#     </div>
# """, unsafe_allow_html=True)

st.html(f"""
    <div style='margin-top: 40px;'>       
        <img src="data:image/png+xml;base64,{get_image(logo_pleias)}">
    </div>
""")