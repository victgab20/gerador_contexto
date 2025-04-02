from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df_humano = pd.read_csv(r"C:\Users\victo\Downloads\train2024 - train2024 (1).csv")
df_llm = pd.read_csv("gisela_llm1.csv")

df_novo_llm = df_llm.groupby(['comentario'])['aspecto_llm'].apply(list).reset_index(name="aspec_list")
df_novo_humano = df_humano.groupby(['texto'])['aspect'].apply(list).reset_index(name="aspec_list")

df_novo_llm_polaridade = df_llm.groupby(['comentario'])['polaridade_llm'].apply(list).reset_index(name="polaridade_list")
df_novo_humano_polaridade = df_humano.groupby(['texto'])['polarity'].apply(list).reset_index(name="polaridade_list")

df_novo_llm = pd.merge(df_novo_llm, df_novo_llm_polaridade, on='comentario', how='left')
df_novo_humano = pd.merge(df_novo_humano, df_novo_humano_polaridade, on='texto', how='left')

y_true = []
y_pred = []
for i in range(len(df_novo_humano)):
    aspectos_referencia = df_novo_humano['aspec_list'][i]
    aspectos_previsoes = df_novo_llm['aspec_list'][i]
    polaridade_referencia = df_novo_humano['polaridade_list'][i]
    polaridade_previsoes = df_novo_llm['polaridade_list'][i]
    for i in range(len(aspectos_referencia)):
        aspectos_previsoes[i]
    
        if aspectos_referencia[i] == aspectos_previsoes[i]:
            print(polaridade_referencia[i])
            y_true.append(polaridade_referencia[i])
            y_pred.append(polaridade_previsoes[i])

print(y_pred)
print(y_true)

labels =[1, -1, 0]

# Exibe matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("Matriz de Confusão")
plt.show()

print(classification_report(y_true, y_pred, target_names=['positivo', 'negativo', 'neutro'], digits=4))
