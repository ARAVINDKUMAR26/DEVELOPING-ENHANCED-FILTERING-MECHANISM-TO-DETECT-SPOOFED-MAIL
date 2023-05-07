import email
import imaplib
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def extract_content(email_message):
    content = ""

    if email_message.is_multipart():
        for part in email_message.walk():
            if part.get_content_type() == 'text/plain':
                content += part.get_payload()
    else:
        content = email_message.get_payload()

    return content


def classify_email(content):
    # Load the trained ML model and vectorizer
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # Preprocess the email content
    content = content.lower()
    content = re.sub('<[^>]+>', '', content)
    content = re.sub('[{}]'.format(re.escape(string.punctuation)), '', content)

    # Vectorize the email content
    content_vector = vectorizer.transform([content])

    # Classify the email as spam or not spam
    prediction = model.predict(content_vector)

    return prediction[0]


def quarantine_spam_mail(username, password, spam_folder, INBOX=None):
    mail = imaplib.IMAP4_SSL('172.20.177.138',993)
    mail.login(username, password)
    mail.select('INBOX')  # Select the INBOX folder

    _, data = mail.search(INBOX, 'ALL')  # Perform search within the selected folder
    email_ids = data[0].split()
    for email_id in email_ids:
        _, data = mail.fetch(email_id, '(RFC822)')
        raw_email = data[0][1]
        email_message = email.message_from_bytes(raw_email)
        content = extract_content(email_message)
        prediction = classify_email(content)

        if prediction == 'spam':
            # Perform the action to quarantine the email
            # For example, move it to a quarantine folder
            mail.copy(email_id, 'Quarantine')
            mail.store(email_id, '+FLAGS', '\\Deleted')

    mail.expunge()
    mail.close()
    mail.logout()


# Provide your email credentials and spam folder name
username = 'devil'
password = '9655329247'
spam_folder = 'Spam'

# Call the quarantine_spam_mail function
quarantine_spam_mail(username, password, spam_folder)

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
