import optuna
from sqlalchemy import create_engine, text

# caminho para o banco de dados
db_path = "pycache/hypertunning.db"
engine = create_engine(f"sqlite:///{db_path}")

# utilizado para armazenamento dos nomes dos estudos
names = []

# Consulta todos os estudos no banco e retorna
with engine.connect() as conn:
    result = conn.execute(text("SELECT study_name FROM studies")).fetchall()
    for row in result:
        names.append(row[0])

# itera entre os estudos para apresentar os melhores valores obtidos
for name in names:
    print('#'*50)
    print('Nome do estudo: ',name)
    study = optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{db_path}"
    )

    print("Melhor trial:")
    print("Número:", study.best_trial.number)
    print("Valor do objetivo (erro ou score):", study.best_trial.value)
    print("Hiperparâmetros:", study.best_trial.params)