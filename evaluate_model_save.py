# evaluate_model_save.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf  # Adicionado para resolver o erro do 'tf'
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.metrics import Precision, Recall, AUC
from keras import backend as K

# ================================
# Funções métricas customizadas
# ================================
def recall_m(y_true, y_pred):
    return Recall()(y_true, y_pred)

def precision_m(y_true, y_pred):
    return Precision()(y_true, y_pred)

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# ================================
# Função principal
# ================================
def main():
    try:
        # 📂 Diretório onde o script está
        RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

        # 📂 Caminho do modelo
        model_path = os.path.join(RESULTS_DIR, "final_model.keras")

        # Verifica se o modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        # 📂 Carregar dados de teste
        X_test = np.load(r"C:\Users\dsimg\Desktop\Projeto\Data\Test set\2D\X_test_2D.npy")
        y_test = np.load(r"C:\Users\dsimg\Desktop\Projeto\Data\Test set\y_test.npy")

        # Ajustar shape se necessário
        if len(X_test.shape) == 3:
            X_test = np.expand_dims(X_test, -1)

        # 🔄 Carregar modelo pré-treinado
        model = load_model(model_path, custom_objects={"f1_m": f1_m})

        # 📊 Avaliar no conjunto de teste
        loss, acc, precision, recall, auc_score = model.evaluate(X_test, y_test, verbose=0)
        print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc_score:.4f}")

        # Predições
        y_pred_proba = model.predict(X_test).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # ================================
        # Salvar resultados em CSV
        # ================================
        results = {
            "loss": loss,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "auc": auc_score,
            "f1_score": float(f1_m(
                tf.convert_to_tensor(y_test, dtype=tf.float32),
                tf.convert_to_tensor(y_pred, dtype=tf.float32)
            ).numpy())
        }
        pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"), index=False)

        # ================================
        # Matriz de Confusão
        # ================================
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0","1"], yticklabels=["0","1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - Loaded Model")
        plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
        plt.close()

        # ================================
        # ROC Curve
        # ================================
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1], [0,1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Loaded Model")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
        plt.close()

        print(f"\nTodos os resultados e gráficos foram salvos em: {RESULTS_DIR}")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# ================================
# Executa apenas se for o script principal
# ================================
if __name__ == "__main__":
    main()
