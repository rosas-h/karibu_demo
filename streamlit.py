import streamlit as st
import pandas as pd

# read csv
df = pd.read_csv('content/question_demo_bsf.csv')
text = df['question'][0]


# logo
st.image("content/Logo_Karibu.png", width=150)

# exercise title
st.markdown("""
    <div style="background-color: #F44A9D; padding: 4px; text-align: left;">
        <p style="color: white; font-family: 'Helvetica';">✍️ EXPRESSION ÉCRITE</p>
    </div>
    """, unsafe_allow_html=True)


# Prompt
st.write("À vous de jouer ! " + text)

# Input field
text_input = st.text_area("Écrivez ici, au moins 30 mots...", height=200)

# Button
if st.button('Valider'):
    if len(text_input) < 30:
        st.warning("Le texte doit contenir au moins 30 mots.")
    else:
        st.success("Texte validé avec succès!")

st.markdown("<style>div.stButton {display: flex; justify-content: center;}</style>", unsafe_allow_html=True)

print(text_input)
