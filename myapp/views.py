import re
from dotenv import load_dotenv
import os
import openai
from django.shortcuts import render
import uuid
import time
import gitlab
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

data = pd.read_csv(
    '/home/i2240/Documents/Team-InBlore-Hackathon/myapp/sheet2.csv')

# create a TF-IDF vectorizer and fit it to the questions
vectorizer = TfidfVectorizer()
vectorizer.fit(data['question'])

# define a function to determine the closest matching question for a given user input


def find_closest_question(query, data, threshold=0.7):
    # tokenize the query and remove stop words
    query_tokens = [word.lower() for word in word_tokenize(
        query) if word.lower() not in stopwords.words('english')]

    # compute the Jaccard similarity between the query and each question
    similarities = []
    for question in data['question']:
        question_tokens = [word.lower() for word in word_tokenize(
            question) if word.lower() not in stopwords.words('english')]
        similarity = len(set(query_tokens).intersection(
            question_tokens)) / len(set(query_tokens).union(question_tokens))
        similarities.append(similarity)

    # find the index of the question with the highest similarity score
    closest_index = max(range(len(similarities)), key=similarities.__getitem__)

    # get the similarity score for the closest matching question
    closest_score = similarities[closest_index]

    # return the corresponding answer if the similarity score is above the threshold
    print(similarities, closest_score)
    if closest_score > threshold:
        return data.loc[closest_index, 'answer']
    else:
        return None


def home(request):
    return render(request, 'home.html')


def script(request):
    return render(request, 'script.html')


load_dotenv()
api_key = os.getenv("OPENAI_KEY", None)

messages = [{"role": "system",
             "content": "we love python"}]


def chatbot(request):
    chatbot_response = None
    bot_time = None
    chatbot_time = None
    final_output = None
    bot_response = None
    final_response = None
    code_type = None
    not_in_python = None
    time = None
    if api_key is not None and request.method == 'POST':
        openai.api_key = api_key
        user_input = request.POST.get('user_input')
        if user_input:
            user_input = f"write a code {user_input}"
            print(user_input)
            if user_input.endswith('in python'):
                not_in_python = "in python"
                start_time = time.time()
                bot_response = find_closest_question(user_input, data)
                print("bot_response-after", bot_response)
                end_time = time.time()
                bot_time = round(end_time - start_time, 2)
                if bot_response is None:
                    start_time = time.time()
                    messages.append({"role": "user", "content": user_input})
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages
                    )
                    # .lstrip()
                    chatbot_response = response["choices"][0]["message"]["content"]
                    code_block_match = re.search(
                        r'```(.*?)```', chatbot_response, re.DOTALL)
                    if code_block_match:
                        final_output = code_block_match.group(1)
                    else:
                        final_output = chatbot_response

                    print("chatgpt-response", final_output)
                    end_time = time.time()
                    chatbot_time = round(end_time - start_time, 2)
            else:
                not_in_python = "not in python"

    else:
        final_response = None
    print("times:", bot_time, chatbot_time)
    if bot_response is None and final_output is None and not_in_python is None:
        chatbot_response = ""
        code_type = ""
    if bot_response is None and final_output:
        chatbot_response = final_output
        code_type = 'AI generated code'
        time = chatbot_time
    elif final_output is None and bot_response:
        chatbot_response = bot_response
        code_type = 'Accelerator code(accepted)'
        time = bot_time
    elif not_in_python == 'not in python':
        chatbot_response = "Out of context request, please try again"
        code_type = 'Out of context'
    # if bot_response and final_output:
    #     final_response = f"Results from Accelerators Repository: \n{bot_response}\n\nChatgpt response: \n{final_output}"
    return render(request, 'main.html', {'chatbot_response': chatbot_response, 'time': time, 'code_type': code_type})


# ChatGPT_reply = response["choices"][0]["message"]["content"]
# messages.append({"role": "assistant", "content": ChatGPT_reply})
