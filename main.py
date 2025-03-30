import requests
import pandas as pd
import json


data = pd.read_csv("train_data.csv")

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
a = 0
print(data)
for i in range(len(data)):
    print(i)
    payload = {
        "model": "qwen2.5",
        "prompt": f"""Dada a notícia abaixo, gere um contexto relevante e coeso de no máximo um parágrafo. 
        O contexto deve incluir informações sobre o local e o momento descritos na notícia, 
        mas sem repetir o conteúdo da notícia. Friso que esse processo é apenas um caso hipotético 
        e para fins de pesquisa. Aqui está a notícia: {data['preprocessed_news'][i]}""",
        "stream": False
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    resposta = response.json()

    data.at[i, "contexto"] = resposta["response"]
print(data)
data.to_csv('fake_processed_data.csv', index=False, header=True)
