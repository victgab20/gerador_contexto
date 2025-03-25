import pandas as pd

data = pd.read_csv(r"C:\Users\victo\Downloads\train2024 - train2024.csv")
# def pre_process(dataframe: pd.DataFrame):
#     dataframe["label"] = dataframe["label"].apply(lambda row: 0 if row == "fake" else 1)

#     return dataframe
# d = pre_process(data)
print(data)
df = data.head(int(len(data) * 0.001))
print(df)
