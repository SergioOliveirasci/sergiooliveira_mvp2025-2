# sergiooliveira_mvp2025-2

💊 Projeto MVP 2025/2 — Machine Learning & Analytics (PUC-Rio)

Previsão de “Alta Aceitação” de Medicamentos a partir de texto clínico

Este repositório contém o projeto desenvolvido como MVP (Mínimo Produto Viável) da Sprint de Machine Learning & Analytics da Pós-Graduação PUC-Rio.

O objetivo é aplicar técnicas de Ciência de Dados e Aprendizado de Máquina para prever se um medicamento terá alta aceitação (definida como Excellent Review % ≥ 50) utilizando apenas informações textuais clínicas e o fabricante — com guard-rails anti-vazamento de alvo.

🎯 Objetivo do Projeto

Definir um problema supervisionado de classificação binária.

Preparar e explorar um dataset real e público (carregado por URL).

Construir pipelines de ML (pré-processamento + modelo) e comparar algoritmos.

Avaliar com métricas adequadas ao desbalanceamento (F1-macro) e discutir trade-offs via tuning de threshold.

Entregar narrativa reprodutível e checklist final.

📂 Estrutura do Repositório

mvp_medicines_high_acceptance.ipynb → Notebook principal (Google Colab).

Medicine_Details.csv → Dataset (carregado por GitHub RAW dentro do notebook).

README.md → Este arquivo.

Observação: o arquivo .csv pode permanecer apenas referenciado por URL (não precisa estar versionado).

🛠️ Tecnologias e Bibliotecas

Ambiente: Python 3.10+ no Google Colab
Principais libs:

pandas, numpy — manipulação/EDA

scikit-learn — ColumnTransformer, Pipeline, TfidfVectorizer, OneHotEncoder, modelos e métricas

matplotlib — curvas ROC e Precision-Recall

🔍 Metodologia (resumo)
Coleta e Preparação

Carga via URL pública (GitHub RAW) direto no Colab.

Sanidade: percentuais em [0,100] e soma≈100%; tratamento de nulos.

Anti-leak: construção de text_all = Composition + Uses + Side_effects e bloqueio de termos de review/rating/score/vote (permitindo % de dosagem).

Pipeline de Pré-processamento

Texto: TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=2).

Categórico: OneHotEncoder(handle_unknown="ignore") para Manufacturer.

Tudo acoplado em ColumnTransformer + Pipeline.

Split, Validação e Modelagem

Split 70/15/15 (treino/val/teste), estratificado.

CV: StratifiedKFold(5) com GridSearchCV (scoring=F1-macro).

Modelos:

Baseline – Logistic Regression

Alternativos – Random Forest, Gradient Boosting

Avaliação

Métrica principal: F1-macro (adequada ao desbalanceamento ~70/30).

Secundárias: Accuracy, ROC-AUC, F1 por classe, PR/AP.

Tuning de threshold (0.10→0.90) para ajustar precisão × recall.

📈 Resultados (principais)

Dados: 11.825 linhas × 9 colunas. Classe positiva ≈ 30,3%.

Validação (melhores por modelo):

RandomForest: F1-macro ≈ 0,557, ROC-AUC ≈ 0,617

Logistic Regression: F1-macro ≈ 0,552 (diferença mínima), ROC-AUC ≈ 0,641

Gradient Boosting: F1-macro ≈ 0,508

Tuning de Threshold (validação):

t ≈ 0,35 → F1-macro ≈ 0,577 (↑ recall, ↓ precisão) vs. t=0,50 (≈0,556)

Teste (final fit em treino+val):

t = 0,50 → F1-macro ≈ 0,560, Accuracy ≈ 0,678
– Classe 1: Precision = 0,445 | Recall = 0,264 | F1 ≈ 0,332

t ≈ 0,35 → F1-macro ≈ 0,560, Accuracy ≈ 0,621
– Classe 1: Precision = 0,382 | Recall = 0,410 | F1 ≈ 0,395

Curvas (teste): ROC-AUC = 0,616 · AP (PR) = 0,402 (> prevalência ≈ 0,303)

Leitura: o tuning move o ponto operacional (mais recall ou mais precisão), mas F1-macro no teste fica estável.
Custo computacional (CV/ fold): LogReg ≈ 0,76 s · GB ≈ 47,8 s · RF ≈ 632 s.
Recomendação de MVP: Logistic Regression (quase mesmo F1, ROC-AUC maior, ordens de grandeza mais barata e interpretável); RF como alternativa.

✅ Checklist de Requisitos (PUC-Rio)

 Dataset público via URL (carga no notebook)

 Problema de ML definido (classificação binária)

 Exploração e tratamento de dados (sanidade + anti-leak)

 Baseline implementado (LogReg)

 Múltiplos modelos comparados (LogReg, RF, GB)

 Pipeline com ColumnTransformer (TF-IDF + OneHot)

 GridSearchCV aplicado com StratifiedKFold(5)

 Avaliação com F1-macro, ROC-AUC, PR/AP, matrizes e threshold tuning

 Conclusões e análise crítica (trade-offs & decisão operacional)

 Checklist final preenchido no notebook

📌 Como Executar
Opção A — Google Colab (recomendado)

Abra o notebook mvp_medicines_high_acceptance.ipynb no Google Colab.

Execute as células na ordem.

O dataset será carregado por URL pública (GitHub RAW) conforme instruções do notebook.

Opção B — Local (Jupyter)
git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
cd <seu-repositorio>
crie e ative o ambiente; instale requirements compatíveis
jupyter notebook mvp_medicines_high_acceptance.ipynb

🧭 Limitações & Próximos Passos

Rótulo heurístico (≥50%) e possível viés de avaliações públicas.

Probabilidades não calibradas (stabilidade do threshold).

Próximos passos:

Calibração (Platt/Isotônica) + novo tuning de threshold;

Refino de TF-IDF (min_df, max_features, char n-grams) e embeddings;

Class weights para aumentar recall com menor queda de precisão;

Explainability (coeficientes/SHAP) + Model Card;

Monitoramento (F1-macro, P/R(+), ROC-AUC/AP, calibração e drift).
