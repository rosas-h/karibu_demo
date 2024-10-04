import streamlit as st
import pandas as pd
import base64
import random
from transformers import pipeline
from ai_tutor import correct_text

token = st.secrets["hf_token"]

@st.cache_data
def get_data():
    return pd.read_csv('content/question_demo_bsf.csv')

# read csv and get questions by level
df = get_data()
beginner = df[df['level'] == 'beginner']['question'].tolist()
intermediate = df[df['level'] == 'intermediate']['question'].tolist()
advanced = df[df['level'] == 'advanced']['question'].tolist()

# model to predict the level of the input text
@st.cache_resource
def get_model():
    return pipeline("sentiment-analysis", model="aapoliakova/cls_level_bsf")

classifier = get_model()

# initialize the session state

if 'text' not in st.session_state:
    st.session_state.text = random.choice(beginner)

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if 'user_text' not in st.session_state:
    st.session_state.user_text = ""

if isinstance(st.session_state.user_input, str):
    st.title(f'{st.session_state.user_input}')





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
@st.cache_data
def get_image(icon_path):
    file_ = open(icon_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url


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
label = "Écrivez un texte d'au moins 30 mots"
if st.session_state.user_input:
    label = st.session_state.user_input

st.text_area(
    label, 
    key="user_text",
    label_visibility="collapsed", 
    placeholder="Écrivez un texte d'au moins 30 mot", 
    height=200
)
st.button('Valider', type="primary", key="main_button")

# Button
if st.session_state.main_button:
# if st.session_state.user_text:
    if len(st.session_state.user_text.split()) < 30:
        st.warning(f"Le texte doit contenir au moins 30 mots")
    else:
        # st.subheader('**:green[Correction]**')
        green_header('Correction')
        
        with st.spinner('Correction en cours, veuillez patienter...'):
        # get correction from the model
            correction = correct_text(st.session_state.user_text, token)
            print(correction)
            st.write(correction)

        # Classifier and recomendation
        green_header('Recommandation')

        # predict the level of the input text
        res = classifier(st.session_state.user_text)[0]['label']
        print("level:", res)

        # get random question based on the level
        if res == 'advanced':
            st.session_state.text = random.choice(advanced)
        elif res == 'intermediate':
            st.session_state.text = random.choice(intermediate)
        else:
            st.session_state.text = random.choice(beginner)
        st.write("Nous avons choisi une nouvelle activité pour vous.  Cliquez ici pour continuer !")
        if st.button("Commencer l'exercice suivant", type="primary"):
            st.session_state["user_text"] = ""
            st.session_state.user_input = ""
            st.rerun()

# st.session_state.user_input = st.session_state.user_text


st.html(f"""
    <div style='margin-top: 10px;'>       
        <img src="data:image/png+xml;base64,{get_image(logo_pleias)}">
    </div>
""")