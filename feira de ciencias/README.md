# MotorSecurity PRO

Sistema de monitoramento de sonolência e distração de motoristas em tempo real, feito em Python.

## O que o projeto faz

O sistema usa a câmera do computador para analisar o rosto do motorista continuamente e identificar sinais de risco durante a direção.

* Sonolência: mede o fechamento dos olhos e dispara alerta quando ficam fechados por tempo excessivo
* Distração: detecta quando o motorista desvia o olhar ou a cabeça da via
* Cansaço: identifica bocejo através da abertura prolongada da boca
* Uso de celular: percebe quando a mão se aproxima da orelha

Quando alguma dessas condições é confirmada, com filtros para evitar falsos positivos, o sistema mostra um alerta visual na tela e avisa por voz.

## Como funciona

A base do projeto é o MediaPipe, um modelo de inteligência artificial do Google treinado para reconhecer rostos e mãos em imagens. Ele é o responsável por localizar, em tempo real, cada ponto do rosto (olhos, boca, contorno) e das mãos do motorista a partir do vídeo capturado pela câmera.

Com esses pontos em mãos, o restante do sistema entra em ação.

* OpenCV processa os frames da câmera e desenha os indicadores visuais
* Um filtro de média móvel suaviza as medições e evita oscilações bruscas
* Um sistema de histerese por tempo garante que o alerta só dispare se a condição se mantiver por alguns segundos, evitando alarme falso por um piscar de olhos comum
* A interface é feita em PyQt5, mostrando o vídeo ao vivo com os indicadores de status
* Os avisos falados usam Edge TTS, com vozes neurais em português, e caem para o pyttsx3 caso não haja internet

## Tecnologias

Python, OpenCV, MediaPipe, PyQt5, Edge TTS, NumPy

## Como instalar

```bash
git clone https://github.com/SEU_USUARIO/motor-security.git
cd motor-security
pip install -r requirements.txt
```

## Como rodar

```bash
python motorsecurity.py
```

Ao abrir, o sistema faz uma calibração rápida, de alguns segundos, medindo o estado normal dos olhos do usuário antes de começar a monitorar.

## Autor

Henrique Barcelos Ribeiro
Estudante de Segurança Cibernética na FIAP. [LinkedIn](https://www.linkedin.com/in/henriquebarcelos144)
