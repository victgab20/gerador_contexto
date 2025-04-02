# Dados de exemplo (aspectos reais e previstos para cada comentário)
true_aspects = [
    {"câmera", "bateria"},      # Comentário 1
    {"tela"},                   # Comentário 2
    {"atendimento"},            # Comentário 3
    {"design", "preço"},        # Comentário 4
    {"entrega"},                # Comentário 5
    {"sistema operacional"},    # Comentário 6
    {"som"},                    # Comentário 7
    {"conexão"},                # Comentário 8
    {"aplicativo"},             # Comentário 9
    {"embalagem"}               # Comentário 10
]

predicted_aspects = [
    {"câmera", "bateria"},      # Comentário 1
    {"tela", "brilho"},         # Comentário 2
    {"atendimento"},            # Comentário 3
    {"design"},                 # Comentário 4
    {"entrega", "custo"},       # Comentário 5
    {"sistema"},                # Comentário 6
    {"som"},                    # Comentário 7
    {"Wi-Fi"},                  # Comentário 8
    {"aplicativo"},             # Comentário 9
    {"embalagem"}               # Comentário 10
]

# Inicializar contadores
TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives

# Calcular TP, FP, FN para cada comentário
for true, pred in zip(true_aspects, predicted_aspects):
    # True Positives: elementos corretamente previstos
    tp = len(true & pred)
    TP += tp
    
    # False Positives: elementos previstos que não estão nos rótulos reais
    fp = len(pred - true)
    FP += fp
    
    # False Negatives: elementos reais não previstos
    fn = len(true - pred)
    FN += fn

# Calcular métricas
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Resultados
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precisão: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1_score:.2%}")