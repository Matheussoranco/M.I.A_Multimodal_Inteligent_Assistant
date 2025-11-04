# M.I.A — Assistente Inteligente Multimodal

> Uma plataforma de IA multimodal que integra linguagem, áudio e visão sob uma arquitetura cognitiva unificada, com memória de longo prazo, automação e API.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/transformers)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green?style=flat-square)](https://chromadb.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-API%20Server-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)


## Visão geral

M.I.A é um assistente inteligente multimodal com foco em:

- Entendimento e geração de linguagem natural (LLM local/remoto)
- Processamento de áudio em tempo real (fala para texto e síntese de voz)
- Processamento de imagens (análise e grounding visual)
- Memória e contexto (vetores e grafo de conhecimento)
- Automação (comandos do agente, execução de ações) e API

Ele oferece uma CLI interativa, componentes desacoplados e um servidor HTTP leve para verificação de saúde. A ideia do projeto é demonstrar uma arquitetura cognitiva prática — que combina raciocínio passo-a-passo com memória e módulos multimodais — e, ao mesmo tempo, ser extensível para casos de uso reais (assistentes de produtividade, copilotos de código, análise multimodal, etc.).


## Problema e motivação

Sistemas de IA práticos precisam ir além de “perguntas e respostas” de texto:

- Unificar múltiplas entradas (voz, imagem, texto) num único fluxo cognitivo.
- Manter memória útil (curto e longo prazo) para personalização e continuidade.
- Conectar ferramentas/ações para executar tarefas no mundo real, com segurança.
- Operar tanto com modelos locais (privacidade) quanto com provedores externos.

O M.I.A aborda isso por meio de uma arquitetura modular com “importações opcionais”, permitindo que cada ambiente carregue apenas o que precisa.


## Objetivos do projeto

- Fornecer um “núcleo cognitivo” simples de integrar e testar.
- Isolar componentes multimodais (áudio/visão) e de memória para evoluir de forma independente.
- Suportar execução local (Ollama/transformers) e remota (OpenAI etc.).
- Disponibilizar uma CLI amigável e uma API mínima para orquestração.


## Princípios de design

- Modularidade e extensibilidade (camadas bem definidas, plugins opcionais)
- “Fail-soft” (funcionar em modo texto mesmo sem visão/áudio/LLM)
- Observabilidade (status, logs enxutos, métricas de performance)
- Portabilidade (Windows/Linux/macOS; CPU/GPU quando possível)


## Arquitetura (alto nível)

```
┌────────────┐   ┌──────────────────┐   ┌──────────────┐
│ Entrada    │ → │ Núcleo Cognitivo │ → │ Camada LLM   │
│ (texto/    │   │ (razão + orques.)│   │ (local/remoto)│
│ áudio/img) │   └──────────────────┘   └──────────────┘
	│                 │                         │
	│                 ↓                         │
	│           Memória/Contexto                │
	│         (vetores + grafo)                 │
	│                 │                         │
	└──────────────→ Ações/Plugins ─────────────┘
```

Componentes principais (diretório `src/mia`):

- `llm/LLMManager`: encapsula consultas a modelos (ex.: Ollama, OpenAI).
- `multimodal/processor` e `vision_processor`: pipeline de visão/imagens.
- `audio/speech_processor` e `speech_generator`: STT/TTS (realtime-friendly).
- `memory/knowledge_graph` e `long_term_memory`: contexto e recordação.
- `plugins/action_executor`: automação (criar arquivo, analisar código, etc.).
- `api/server.py`: servidor FastAPI minimalista com `/health` e `/ready`.
- `performance_monitor`, `cache_manager`, `resource_manager`: suporte.


## Fluxos de uso

1) Texto → LLM → resposta (com consulta opcional à memória)
2) Áudio → (captura + transcrição) → LLM → síntese de voz
3) Imagem → (pré-processamento/visão) → LLM com grounding visual
4) Comandos do agente → executor de ações (ex.: criar arquivo, analisar código)

A CLI admite comandos como `help`, `status`, `clear`, `models`, `quit` e comandos do agente (ex.: “create file notas.txt”).

### "Jarvis" — capacidades de automação (atual/planejado)

O M.I.A inclui um executor de ações (`ActionExecutor`) que concentra automações úteis — algumas já disponíveis, outras planejadas — para aproximar a experiência de um "Jarvis":

- Mensageria
	- WhatsApp (via `pywhatkit` → WhatsApp Web no PC) — atual: envio imediato
	- Telegram (via Bot API ou Telegram Desktop por automação) — atual (via Bot API)
	- SMS (via provedor externo, p.ex. Twilio) — planejado
- E‑mail
	- Envio SMTP (Gmail/SMTP) — atual
- Web
	- Busca e raspagem simples (`web_search`, `web_scrape`) — atual
	- Automação via Selenium (`web_automation`) — atual (requer driver/GUI)
- Arquivos e produtividade
	- Criar/ler/escrever planilhas `.xlsx`/`.csv` (OpenPyXL/CSV) — atual
	- Criar apresentações PowerPoint `.pptx` (python-pptx) — atual
	- Criação simples de documentos (ex.: Markdown) — atual (notas)
	- Geração de código inicial (Python/JS/Java/HTML/CSS) — atual
	- Análise básica de código (contagem de linhas, linguagem) — atual
- Smart home
	- Gatilhos e stubs para Home Assistant — atual (requer URL/TOKEN)
- Sistema
	- Notificações locais, área de transferência, abrir apps, rodar comandos — atual
- Voz
	- Entrada por voz (Whisper/Google SR) — atual (dependências opcionais)
	- Saída por voz (TTS via APIs: OpenAI/Minimax/Nanochat) — atual; TTS local desativado por padrão
- "Máquina virtual" interna
	- Planejado: sandbox/tarefas isoladas (WASM/WASI ou microVM/container) com permissões e orçamento de recursos

> Observação: alguns módulos exigem dependências de SO, GUI e/ou credenciais. Em ambientes headless, use `opencv-python-headless` e ajuste permissões.


## Estado do projeto

- Versão atual: `0.1.0-alpha.1` (status: Alpha)
- Foco atual: prova de conceito estável em modo texto e estruturação da arquitetura.
- Módulos multimodais e de automação estão disponíveis de forma **opcional** e podem exigir dependências específicas do SO.


## Requisitos

- Python 3.8+
- Recomendado: PyTorch 2.0+ (CPU ou GPU)
- Para LLM local: Ollama ou modelos via `transformers`
- Para memória vetorial: ChromaDB
- Para áudio: PortAudio/sounddevice; para TTS/STT, libs específicas
- Para visão: Pillow; (OpenCV pode ser necessário em alguns fluxos)

> Observação (Windows): em ambientes sem GUI ou com restrições, prefira `opencv-python-headless` e avalie fixar `numpy<2` para compatibilidade com algumas libs nativas.


## Instalação

### Clonar e instalar dependências (modo simples)

```powershell
git clone https://github.com/Matheussoranco/M.I.A-The-successor-of-pseudoJarvis.git
cd M.I.A-The-successor-of-pseudoJarvis
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dicas de compatibilidade (opcional)

```powershell
# Em servidores/CI ou Windows sem GUI, pode ajudar:
pip install "numpy<2" opencv-python-headless
```

### Suporte a LLM local (Ollama) — opcional

- Instale o Ollama e baixe um modelo:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:1.5b
```


## Configuração

Crie um arquivo `.env` (ou use `config/.env.example` como referência):

```env
# Modelos locais (Ollama)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:1.5b

# Provedores remotos (quando em uso)
# OPENAI_API_KEY=...

# Ajustes gerais
DEBUG_MODE=false

# Comunicação
WHATSAPP_PHONE=+5511999999999
TELEGRAM_BOT_TOKEN=xxxxxxxx:yyyyyyyy

# Email (SMTP)
EMAIL_USERNAME=seuemail@gmail.com
EMAIL_PASSWORD=sua_senha_ou_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Smart home
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=seu_token

# TTS (opcionais)
OPENAI_API_KEY=...
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=alloy
OPENAI_TTS_FORMAT=mp3
```


## Como usar (CLI)

```powershell
# Informações da versão e ambiente
python .\main.py --info

# Modo texto
python .\main.py --mode text

# Modo áudio (requer microfone e deps de áudio)
python .\main.py --mode audio

# Modo misto (padrão)
python .\main.py --mode mixed

# Alterar modelo (exemplo com outro ID)
python .\main.py --model-id gemma3:4b-it-qat
```

Comandos interativos na CLI:

- `help` — ajuda rápida
- `status` — status do sistema (LLM, áudio, device, cache)
- `models` — lista de modelos conhecidos
- `clear` — limpa contexto e otimiza caches
- `quit` — sair

Comandos do agente (exemplos):

- `create file notas.txt`
- `analyze code src/mia/main.py`
- `send whatsapp "+5511999999999" "Olá! Mensagem teste."`
- `enviar email para exemplo@dominio.com assunto "Orçamento" corpo "Segue em anexo..."`
- `abrir https://example.com`
- `criar planilha relatorio.xlsx`
 - `send telegram para 123456789 "Olá do M.I.A"`
 - `criar apresentação briefing.pptx "Estratégia 2026"`

### Exemplos programáticos (ActionExecutor)

```python
from mia.tools.action_executor import ActionExecutor

agent = ActionExecutor()
agent.execute("send_whatsapp", {"recipient": "+5511999999999", "message": "Olá!"})
agent.execute("send_email", {"to": "exemplo@dominio.com", "subject": "Oi", "body": "Tudo bem?"})
agent.execute("create_sheet", {"filename": "dados.xlsx", "data": [["A","B"],[1,2]]})
agent.execute("web_automation", {"url": "https://example.com"})
agent.execute("create_presentation", {"filename": "demo.pptx", "title": "Plano 2026", "content": "Resumo executivo"})
```


## API (servidor mínimo)

O servidor expõe endpoints de saúde e prontidão.

```powershell
# Iniciar via entrypoint do pacote (se instalado com extras de API)
mia-api

# ou iniciar o módulo diretamente
python -m mia.api.server
```

Endpoints:

- `GET /health` → `{ "status": "ok", "version": "..." }`
- `GET /ready`  → `{ "status": "ready" }`

Variáveis de ambiente:

- `MIA_API_HOST` (padrão: `0.0.0.0`)
- `MIA_API_PORT` (padrão: `8080`)


## Testes

```powershell
# Executar testes
pytest tests/

# Cobertura
pytest --cov=src tests/
```


## Roadmap (resumo)

- [ ] Lazy import para módulos pesados (visão/automação) para melhorar `--info` e compatibilidade
- [ ] Separar dependências em core/dev/api/audio/vision com arquivos de requisitos claros
- [ ] Modo áudio mais robusto (tratamento de queda de microfone e fallback)
- [x] Memória: consolidação entre vetores e grafo + ferramentas de depuração
- [ ] Expandir API (rotas de chat e de visão) mantendo o servidor minimalista por padrão
- [ ] Docker images “text-only” e “full”
- [ ] Sandbox/VM interna (WASM/WASI ou microVM/container) com limites de recursos e permissões
- [ ] Agente: ampliar parser de comandos (WhatsApp/email/web/sheets) e confirmar ações sensíveis (consentimento)


## Limitações e notas

- Alguns módulos opcionais (ex.: automação com `pyautogui`/`opencv`) exigem GUI e podem não funcionar bem em servidores headless sem ajustes.
- Dependências de áudio/visão variam entre SOs; verifique as dicas de compatibilidade.
- Em status “Alpha”, pode haver mudanças de API/CLI sem aviso em versões iniciais.


## Licença

AGPL-3.0-or-later — consulte `LICENSE`.


## Autoria

Matheus Pullig Soranço de Carvalho — matheussoranco@gmail.com

Se este projeto te ajudou, considere abrir uma issue com feedback ou sugestões.