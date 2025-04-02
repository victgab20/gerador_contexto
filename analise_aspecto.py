import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

df_humano = pd.read_csv(r"C:\Users\victo\Downloads\train2024 - train2024 (1).csv")
df_llm = pd.read_csv("gisela_llm1.csv")

df_novo_llm = df_llm.groupby(['comentario'])['aspecto_llm'].apply(list).reset_index(name="aspec_list")
df_novo_humano = df_humano.groupby(['texto'])['aspect'].apply(list).reset_index(name="aspec_list")

df_novo_llm_polaridade = df_llm.groupby(['comentario'])['polaridade_llm'].apply(list).reset_index(name="polaridade_list")
df_novo_humano_polaridade = df_humano.groupby(['texto'])['polarity'].apply(list).reset_index(name="polaridade_list")

df_novo_llm = pd.merge(df_novo_llm, df_novo_llm_polaridade, on='comentario', how='left')
df_novo_humano = pd.merge(df_novo_humano, df_novo_humano_polaridade, on='texto', how='left')

print(df_novo_humano.head())

TP = 0
FP = 0
FN = 0

for _, row in df_novo_humano.iterrows():
    comentario_humano = row['texto']
    aspectos_reais = set(row['aspec_list'])
    
    comentario_llm = df_novo_llm[df_novo_llm['comentario'] == comentario_humano]
    
    if not comentario_llm.empty:
        aspectos_previstos = set(comentario_llm['aspec_list'].values[0])
        
        tp = len(aspectos_reais & aspectos_previstos)
        TP += tp

        fp = len(aspectos_previstos - aspectos_reais)
        FP += fp
        
        fn = len(aspectos_reais - aspectos_previstos)
        FN += fn

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precisão: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1_score:.2%}")



conf_matrix = pd.DataFrame([
    [TP, FP],
    [FN, 0]  # sem TN nesse caso, pois só temos aspectos positivos
], columns=['Previsto como aspecto', 'Previsto como não aspecto'],
   index=['Realmente é aspecto', 'Realmente não é aspecto'])

# Plot
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title("Matriz de Confusão - Extração de Aspectos", fontsize=14)
plt.xlabel("Predição")
plt.ylabel("Valor Real")

# Adiciona métricas no canto inferior
plt.figtext(0.5, -0.1, f"Precisão: {precision:.2%}  |  Recall: {recall:.2%}  |  F1-Score: {f1_score:.2%}", 
            wrap=True, horizontalalignment='center', fontsize=12)

plt.tight_layout()
plt.show()