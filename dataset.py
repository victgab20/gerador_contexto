import pandas as pd

data = pd.read_csv(r"C:\Users\victo\Downloads\Fake.br-Corpus-master\Fake.br-Corpus-master\preprocessed\pre-processed.csv")
def pre_process(dataframe: pd.DataFrame):
    dataframe["label"] = dataframe["label"].apply(lambda row: 0 if row == "fake" else 1)

    return dataframe
d = pre_process(data)
print(d["preprocessed_news"][0])
