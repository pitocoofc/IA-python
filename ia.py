import numpy as np
import os
import re

# CONFIGURAÇÕES PARA SAIR DA IDIOTICE
FILE_TEXTO = "texto.txt"
FILE_BRAIN = "cerebro_estavel.npy"
HIDDEN_SIZE = 128  # Foco total na estabilidade
LEARNING_RATE = 0.001 # Passo preciso para evitar picos de Loss

def limpar_texto(t):
    return re.sub(r'[^\w\s]', '', t).lower()

def carregar_dados():
    if not os.path.exists(FILE_TEXTO):
        print("❌ Cadê o texto.txt?")
        exit()
    with open(FILE_TEXTO, 'r', encoding='utf-8', errors='ignore') as f:
        words = limpar_texto(f.read()).split()
    return words, sorted(list(set(words)))

palavras_dataset, vocab = carregar_dados()
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# PESOS COM INICIALIZAÇÃO DE XAVIER
if os.path.exists(FILE_BRAIN):
    brain = np.load(FILE_BRAIN, allow_pickle=True).item()
    Wxh, Whh, Why = brain['Wxh'], brain['Whh'], brain['Why']
    bh, by = brain['bh'], brain['by']
else:
    Wxh = np.random.randn(HIDDEN_SIZE, vocab_size) / np.sqrt(vocab_size)
    Whh = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) / np.sqrt(HIDDEN_SIZE)
    Why = np.random.randn(vocab_size, HIDDEN_SIZE) / np.sqrt(HIDDEN_SIZE)
    bh, by = np.zeros((HIDDEN_SIZE, 1)), np.zeros((vocab_size, 1))

def treinar(epochs=200):
    global Wxh, Whh, Why, bh, by
    print(f"🔥 Treinando NDJ-BALÃO... Vocabulário: {vocab_size}")
    for e in range(1, epochs + 1):
        h = np.zeros((HIDDEN_SIZE, 1))
        loss = 0
        for t in range(len(palavras_dataset)-1):
            x = np.zeros((vocab_size, 1))
            x[word_to_ix[palavras_dataset[t]]] = 1
            target = word_to_ix[palavras_dataset[t+1]]
            h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
            y = np.dot(Why, h) + by
            ps = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
            loss += -np.log(ps[target, 0] + 1e-12)
            dy = np.copy(ps); dy[target] -= 1
            dWhy = np.dot(dy, h.T)
            dh = np.dot(Why.T, dy) * (1 - h * h)
            dWxh = np.dot(dh, x.T)
            dWhh = np.dot(dh, h.T) 
            for p, dp in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dh, dy]):
                p -= LEARNING_RATE * np.clip(dp, -1, 1)
        if e % 10 == 0:
            print(f"🎈 Época {e}/{epochs} | Perda: {loss/len(palavras_dataset):.4f}")
    np.save(FILE_BRAIN, {'Wxh': Wxh, 'Whh': Whh, 'Why': Why, 'bh': bh, 'by': by})
    print("✅ Cérebro salvo!")

def conversar(seed, tam=10, temp=0.6):
    palavras_entrada = limpar_texto(seed).split()
    if not palavras_entrada: return ""
    palavra = palavras_entrada[-1]
    if palavra not in word_to_ix: return "Palavra nova..."
    
    res = [palavra]
    h = np.zeros((HIDDEN_SIZE, 1))
    for _ in range(tam):
        x = np.zeros((vocab_size, 1))
        x[word_to_ix[palavra]] = 1
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        ps = np.exp((y - np.max(y)) / temp) / np.sum(np.exp((y - np.max(y)) / temp))
        idx = np.random.choice(range(vocab_size), p=ps.ravel())
        palavra = ix_to_word[idx]
        res.append(palavra)
    
    # VARIÁVEL DE SAÍDA: O navegador vai ler daqui
    resposta_ia = " ".join(res)
    return resposta_ia

# INTERFACE QUE UNE TERMINAL E FUTURO HTML
if __name__ == "__main__":
    # Se rodar direto, ele pergunta. Se for chamado por script, ele só processa.
    print("\n🎈 NDJ-BALÃO v1.5")
    op = input("(t)reinar ou (c)onversar? ")

    if op.lower() == 't':
        treinar(int(input("Épocas: ") or 300))
    else:
        while True:
            pergunta = input("\nVocê: ")
            if pergunta.lower() == 'sair': break
            
            # Lógica: Variável primeiro, print depois
            resultado = conversar(pergunta, 8, 0.6)
            
            # O print é o que o seu sistema de transmissão vai "escutar"
            print(f"{resultado}")
