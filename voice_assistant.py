import speech_recognition as sr
import pyttsx3
import wikipedia
import keyboard

# Configurações de pesquisa na Wikipedia
wikipedia.set_lang("pt")  # Defina o idioma para português

def reconhecer_comando():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Diga algo...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            comando = recognizer.recognize_google(audio, language="pt-BR")
            print(f"Você disse: {comando}")
            return comando.lower()
        except sr.UnknownValueError:
            print("Não entendi o que você disse.")
            return None
        except sr.RequestError:
            print("Erro ao acessar o serviço de reconhecimento de voz.")
            return None

def responder_comando(comando):
    engine = pyttsx3.init()
    
    respostas = {
        "como estou": "Você parece estar bem!",
        "qual meu humor": "Parece que você está feliz!",
        "o que posso melhorar": "Tente sorrir mais para parecer mais confiante!",
    }

    # Verifica se a resposta já está no dicionário
    resposta = respostas.get(comando)

    if not resposta:  # Caso não esteja no dicionário, fazemos uma busca na Wikipédia
        try:
            print("Pesquisando na web...")
            resultado = wikipedia.summary(comando, sentences=1)  # Resumo da pesquisa
            resposta = resultado
        except wikipedia.exceptions.DisambiguationError as e:
            resposta = "Desculpe, sua pergunta está muito vaga. Tente reformular."
        except wikipedia.exceptions.HTTPTimeoutError:
            resposta = "Desculpe, houve um problema ao acessar a Wikipedia."
    
    print(resposta)
    engine.say(resposta)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        if keyboard.is_pressed("q"):  # Verifica se a tecla "q" foi pressionada
            print("Saindo...")
            break  # Sai do loop e finaliza o programa

        comando = reconhecer_comando()
        if comando:
            responder_comando(comando)
