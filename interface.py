import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from PIL import Image, ImageTk

# Função para fazer previsões
def predict():
    try:
        # Obter entradas do usuário
        inputs = {
            'GENDER': gender_var.get(),
            'AGE': int(age_var.get()),
            'SMOKING': int(smoking_var.get()),
            'YELLOW_FINGERS': int(yellow_fingers_var.get()),
            'ANXIETY': int(anxiety_var.get()),
            'PEER_PRESSURE': int(peer_pressure_var.get()),
            'CHRONIC DISEASE': int(chronic_disease_var.get()),
            'WHEEZING': int(wheezing_var.get()),
            'ALCOHOL CONSUMING': int(alcohol_consuming_var.get()),
            'COUGHING': int(coughing_var.get()),
            'SHORTNESS OF BREATH': int(shortness_breath_var.get()),
            'SWALLOWING DIFFICULTY': int(swallowing_difficulty_var.get()),
            'CHEST PAIN': int(chest_pain_var.get()),
        }

        input_df = pd.DataFrame([inputs])

        # Definir as features esperadas pelo modelo
        expected_features = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                             'PEER_PRESSURE', 'CHRONIC DISEASE',
                             'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
                             'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

        # **Tratar features faltantes**
        if expected_features:
            for feature in expected_features:
                input_df[feature] = 1  # Substituir por um valor padrão adequado
            print(f"As seguintes features foram adicionadas com valor padrão: {expected_features}")

        # **Tratar features extras**
        
        # Carregar o modelo treinado
        model = joblib.load('especialista_em_cancer.pkl')

        # Fazer a previsão
        prediction = model.predict(input_df)

        # Mostrar resultado
        result = "YES" if prediction == 0 else "NO"
        messagebox.showinfo("Prediction", f"Seu resultado é: {result}. Câncer no pulmão")

    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("especialista_cp")
root.configure(bg="#a6a6a6")


icone = Image.open("edson_desenvolvedor.ico")  # Substitua pelo caminho do seu ícone
icone_tk = ImageTk.PhotoImage(icone)

        # Definir o ícone da janela
root.iconphoto(False, icone_tk)
        
imagem = tk.PhotoImage(file="edson_desenvolvedor.png")
w = tk.Label(root, image=imagem,height=170, width=310)
w.imagem = imagem
w.place(x=1, y=380)


# Variáveis de entrada
gender_var = tk.StringVar()
age_var = tk.StringVar()
smoking_var = tk.StringVar()
yellow_fingers_var = tk.StringVar()
anxiety_var = tk.StringVar()
peer_pressure_var = tk.StringVar()
chronic_disease_var = tk.StringVar()
wheezing_var = tk.StringVar()
alcohol_consuming_var = tk.StringVar()
coughing_var = tk.StringVar()
shortness_breath_var = tk.StringVar()
swallowing_difficulty_var = tk.StringVar()
chest_pain_var = tk.StringVar()

# Texto explicativo no topo
info_label = tk.Label(root, text="Genero: 0 = M, 1 = F", fg="blue")
info_label.grid(row=0, column=0, columnspan=2, pady=(5, 10))

# Criar labels e campos de entrada
fields = [
    ("Gênero (M/F):", gender_var),
    ("Idade:", age_var),
    ("Fumar (1/0):", smoking_var),
    ("Dedos Amarelos (1/0):", yellow_fingers_var),
    ("Ansiedade (1/0):", anxiety_var),
    ("Pressão dos Pares (1/0):", peer_pressure_var),
    ("Doença Crônica (1/0):", chronic_disease_var),
    ("Chiado (1/0):", wheezing_var),
    ("Consumir Álcool (1/0):", alcohol_consuming_var),
    ("Tosse (1/0):", coughing_var),
    ("Falta de Ar (1/0):", shortness_breath_var),
    ("Dificuldade de Engolir (1/0):", swallowing_difficulty_var),
    ("Dor no Peito (1/0):", chest_pain_var),
]

for i, (label_text, var) in enumerate(fields):
    label = tk.Label(root, text=label_text)
    label.grid(row=i + 2, column=0, padx=10, pady=5, sticky=tk.W)
    entry = tk.Entry(root, textvariable=var)
    entry.grid(row=i + 2, column=1, padx=10, pady=5)

# Texto explicativo abaixo da idade
age_note_label = tk.Label(root, text="1 = Sim, 0 = Não", fg="blue")
age_note_label.grid(row=1, column=0, columnspan=2, pady=(5, 10))

# Botão de previsão
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=len(fields) + 2, column=0, columnspan=2, pady=20)

# Executar a interface
tk.mainloop()



