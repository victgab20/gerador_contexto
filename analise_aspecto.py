import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

df_humano = pd.read_csv(r"C:\Users\victo\Downloads\train2024 - train2024 (1).csv")
df_llm = pd.read_csv("gisela_llm1.csv")

def limpar(texto):
    return str(texto).strip().lower()

df_humano["texto"] = df_humano["texto"].apply(limpar)
df_humano["aspect"] = df_humano["aspect"].apply(limpar)
df_llm["comentario"] = df_llm["comentario"].apply(limpar)
df_llm["aspecto_llm"] = df_llm["aspecto_llm"].apply(limpar)

aspectos_humano = df_humano.groupby("texto")["aspect"].apply(lambda x: list(set(x))).reset_index()
aspectos_llm = df_llm.groupby("comentario")["aspecto_llm"].apply(lambda x: list(set(x))).reset_index()

df_aspectos = pd.merge(aspectos_humano, aspectos_llm, left_on="texto", right_on="comentario", how="inner")

mlb = MultiLabelBinarizer()
y_true_bin = mlb.fit_transform(df_aspectos["aspect"])
y_pred_bin = mlb.transform(df_aspectos["aspecto_llm"])

precision = precision_score(y_true_bin, y_pred_bin, average='micro')
recall = recall_score(y_true_bin, y_pred_bin, average='micro')
f1 = f1_score(y_true_bin, y_pred_bin, average='micro')
jaccard = jaccard_score(y_true_bin, y_pred_bin, average='micro')

print("🔍 MÉTRICAS DE EXTRAÇÃO DE ASPECTOS (Multilabel por comentário)")
print(f"Precision (micro): {precision:.2f}")
print(f"Recall    (micro): {recall:.2f}")
print(f"F1-score  (micro): {f1:.2f}")
print(f"Jaccard   (micro): {jaccard:.2f}")
