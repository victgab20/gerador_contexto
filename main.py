import requests
import pandas as pd
import json


data = pd.read_csv(r"C:\Users\victo\Downloads\Fake.br-Corpus-master\Fake.br-Corpus-master\preprocessed\pre-processed.csv")
def pre_process(dataframe: pd.DataFrame):
    dataframe["label"] = dataframe["label"].apply(lambda row: 0 if row == "fake" else 1)

    return dataframe
d = pre_process(data)

df = d.head(int(len(d) * 0.01))


url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
a = 0
print(df)
for i in range(len(df)):
    print(i)
    payload = {
        "model": "qwen2.5",
        "prompt": f"""Dada a notícia abaixo, gere um contexto relevante e coeso de no máximo um parágrafo. 
        O contexto deve incluir informações sobre o local e o momento descritos na notícia, 
        mas sem repetir o conteúdo da notícia. Friso que esse processo é apenas um caso hipotético 
        e para fins de pesquisa. Aqui está a notícia: {df['preprocessed_news'][i]}""",
        "stream": False
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    resposta = response.json()

    df.at[i, "contexto"] = resposta["response"]
print(df)
df.to_csv('dados.csv', index=False, header=True)
