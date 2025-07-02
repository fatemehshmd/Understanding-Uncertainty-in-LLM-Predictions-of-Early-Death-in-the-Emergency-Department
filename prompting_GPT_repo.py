print('start')
import pandas as pd
import numpy as np
import openai
import tiktoken
import re
from scipy import spatial
import random

#########################
from openai import OpenAI
#############################################
## Variables
api_key = 'Put your API Key here'

baseURL = 'https://api.openai.com/v1/embeddings'
ChatGPT_baseURL = 'https://api.openai.com/v1/chat/completions'
GPT_MODEL = "gpt-4o"

#############################################
##functions


def split_text(s):
    pattern = re.compile(r'\n\s*(?=[A-Za-z ]+:)', re.MULTILINE)
    result = re.split(pattern,s)

    # Print the result
    split =[]
    for part in result:
        part = re.sub(r"<ref.*?</ref>", "", part)
        part = part.strip()
#         print(part)
        split.append(part)
#         print('\n','*'*50)
    return split



def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_GPT(m,temp):
    message = [{"role": "user", "content": m}]
    response = client.chat.completions.create(model=GPT_MODEL, messages=message, temperature= temp, max_tokens=12, logprobs = True, top_logprobs = 10 )
    
    # Extract the list of top logprobs from the first choice's logprobs content
    top_logprobs_list = response.choices[0].logprobs.content[0].top_logprobs  # Access top logprobs

    # Extract the first 5 tokens and their log probabilities
    top_10_logprobs = [(item.token, item.logprob) for item in top_logprobs_list[:10]]
    return top_10_logprobs




## Main
if  __name__== '__main__':

    client = OpenAI(api_key=api_key)
    print('all well')

    # df = pd.read_excel('df_final_sample.xlsx').drop('Unnamed: 0', axis=1)
    df = pd.read_excel('df_filtered.xlsx').drop('Unnamed: 0', axis=1)




    ## preprocessing
    print(len(df))
    print(df.groupby('HOSPITAL_EXPIRE_FLAG').size())

    prompt = " You are a clinical risk prediction model. I will provide you with a clinical note recorded on the first day of a patient's hospital admission. Based solely on the information provided in the note, your task is to predict whether the patient is at high risk of in-hospital death during their current admission. please carefully analyze the content of the note for clinical indicators such as vital signs, symptoms, laboratory results, and any other documented clinical findings that might signal a critical condition. Your response must be exactly one word: 'Yes' if the note suggests that the patient is at risk of in-hospital death, or 'No' if it does not. Do not include any additional text or explanation.  "


    # sampling rate
    sampling_rate = 30
    output = df['prediction_logprob']
    print(output)
    for i in range(0, len(df)):
        if i>=0:
            print('*' * 50, '\n', 'note number: ', i, '\n')
            print('mortality: ', df['HOSPITAL_EXPIRE_FLAG'].iloc[i])
            print('text length: ', df['len_text_2'].iloc[i])
            output_sample = [0] * sampling_rate
            for j in range(sampling_rate):
                # txt = df['TEXT'].iloc[i]
                txt = df['FIRST_24H_TEXT'].iloc[i]
                # print(txt)


                
                ## query GPT

                # Generate a random float between 0.9 and 1.0
                random_number = random.uniform(0.9, 1.0)
                m = prompt + ' ' + txt
                res = query_GPT(m,random_number)
                # print(res)
                output_sample[j]= res
            output.iloc[i] = output_sample
            # print(output)


            ## save results
            df['prediction_logprob'] = output
            df.to_excel('df_filtered_gpt_temp_1.xlsx')

    ##final saving
    df['prediction_logprob'] = output
    df.to_excel('df_filtered_gpt_temp_1.xlsx')
    print('*********Done**********')


