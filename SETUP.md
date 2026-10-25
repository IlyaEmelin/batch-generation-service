# Развёртывание Qwen2.5-1.5B-Instruct на vLLM в WSL2 (Ubuntu) под Windows

Пошаговая инструкция: установка, запуск, проверка, **остановка** и **полное удаление**.
Все команды проверены на стенде этого проекта 2026-10-23.

Нужен только копипаст — сразу к разделу 11 «Чек-лист с нуля».

Что получилось на стенде:

| Компонент | Значение |
|---|---|
| ОС | Windows 10/11 x64 + WSL2, дистрибутив `Ubuntu-22.04` (Ubuntu 22.04.5 LTS, ядро 5.15.167.4-microsoft-standard-WSL2) |
| Python в WSL | 3.10.12, окружение `/home/user/vllm-env` (вне проекта) |
| vLLM | 0.31.0 (тянет PyTorch 2.13.0, transformers 5.17.0, flashinfer-python 0.7.0.post1, nvidia-cuda-nvcc 13.4.92) |
| Модель | `Qwen/Qwen2.5-1.5B-Instruct`, кэш `~/.cache/huggingface` (2.9 ГБ) |
| GPU | NVIDIA GeForce RTX 2060, 6144 MiB, драйвер Windows 610.62 |
| Порт | 8000 (внутри WSL; из Windows виден как `http://127.0.0.1:8000`) |

## 0. Схема и почему WSL

```
Windows: клиент (python, openai)  --HTTP-->  127.0.0.1:8000
                                              |
                                              v
                          WSL2 Ubuntu-22.04: vllm serve Qwen/Qwen2.5-1.5B-Instruct
                                              |
                                              v
                                        RTX 2060 (6 ГБ)
```

vLLM не поддерживает нативный Windows (нет официальных колёс и ядра), поэтому сервер живёт в Ubuntu под WSL2, а проект может оставаться на диске `C:` — WSL видит его как `/mnt/c/...`.

## 1. Что понадобится

- Windows 10 21H2+ или Windows 11, 64-бит; в BIOS включена виртуализация (VT-x/AMD-V).
- NVIDIA GPU с ≥6 ГБ памяти (здесь RTX 2060 6 ГБ; 1.5B в float16 занимает ~3 ГБ).
- Драйвер NVIDIA для Windows (здесь 610.62). Внутрь WSL драйвер **не** ставится.
- ~20 ГБ свободного места на системном диске (см. таблицу в разделе 9).
- Интернет для скачивания колёс vLLM (~10 ГБ) и весов модели (2.9 ГБ).

## 2. Установка WSL2 и Ubuntu 22.04

В PowerShell **от администратора**:

```powershell
wsl --status                 # есть ли WSL и какая версия по умолчанию
wsl --update                 # обновить ядро WSL2
wsl --set-default-version 2  # новые дистрибутивы — на WSL2
wsl --install -d Ubuntu-22.04
```

При первом запуске Ubuntu попросит создать пользователя и пароль (на стенде — `user`, домашний каталог `/home/user`).

Проверка:

```powershell
wsl -l -v
#   NAME            STATE     VERSION
# * Ubuntu-22.04    Running   2
```

Дальше все команды выполняются **внутри Ubuntu**. Два равнозначных способа:

```powershell
wsl -d Ubuntu-22.04            # открыть терминал Ubuntu
```

```powershell
wsl -d Ubuntu-22.04 -e bash -lc "команда"   # выполнить одну команду в Ubuntu из PowerShell
```

> Если Ubuntu уже установлена под именем `Ubuntu`, подставляйте это имя в `-d`: `wsl -d Ubuntu ...`. Имя видно в `wsl -l -v`.

## 3. Проверка GPU внутри WSL

```bash
nvidia-smi
```

Ожидаемо: `NVIDIA GeForce RTX 2060`, `6144MiB`, `Driver Version: 610.62`. Это тот же драйвер Windows, транслированный в WSL.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# NVIDIA GeForce RTX 2060, 610.62, 6144 MiB
```

Если `nvidia-smi: command not found` или GPU не виден:

1. `wsl --update` из PowerShell, затем перезагрузка Windows;
2. убедиться, что дистрибутив на WSL2 (`wsl -l -v`, столбец VERSION = 2);
3. обновить драйвер NVIDIA **в Windows** (не внутри WSL — установка драйвера в WSL ломает проброс GPU).

CUDA Toolkit внутри WSL ставить не нужно: библиотеки CUDA приходят внутри pip-колёс PyTorch, а `nvcc` — пакетом `nvidia-cuda-nvcc`.

## 4. Python-окружение и vLLM

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
python -m pip install --upgrade pip
pip install vllm==0.31.0
```

`vllm` сам притянет нужные версии `torch`, `transformers`, `flashinfer-python`, `nvidia-cuda-nvcc` — отдельно их ставить не надо. Скачивается ~10 ГБ, окружение занимает 9.9 ГБ.

Проверка:

```bash
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
# 0.31.0 2.13.0 True
```

Почему окружение вне проекта (`~/vllm-env`, а не `.venv` в репозитории): каталог `/mnt/c` медленный (импорт vLLM с него заметно дольше) и смонтирован без прав на `chmod`, поэтому bash-скрипты запускаются через `bash script.sh`, а не как исполняемые файлы.

## 5. Скачивание модели

```bash
source ~/vllm-env/bin/activate
hf download Qwen/Qwen2.5-1.5B-Instruct
```

Веса (~2.9 ГБ) лягут в `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct`. Скрипт запуска работает с `HF_HUB_OFFLINE=1`, то есть модель обязана быть в кэше заранее — иначе сервер упадёт на загрузке.

Проверка:

```bash
du -sh ~/.cache/huggingface          # ~2.9G
```

## 6. Запуск сервера

### Вариант А — отдельное окно Ubuntu, сервер на переднем плане (рекомендуется)

```powershell
wsl -d Ubuntu-22.04 -e bash /mnt/c/PythonProg/batch-generation-service/scripts/start_vllm.sh
```

или внутри Ubuntu:

```bash
cd /mnt/c/PythonProg/batch-generation-service
bash scripts/start_vllm.sh
```

Скрипт `scripts/start_vllm.sh` делает всё сам:

1. активирует окружение `~/vllm-env` (если его нет — подскажет создать);
2. добавляет в `PATH` каталог с `nvcc` из `site-packages/nvidia/cu*/bin` (нужно для CUDA graphs);
3. ставит `HF_HUB_OFFLINE=1`, чтобы старт не зависел от сети;
4. печатает время старта для сверки с отчётом;
5. запускает `vllm serve` с параметрами стенда.

Параметры можно переопределять переменными окружения:

| Переменная | По умолчанию | Зачем менять |
|---|---|---|
| `VENV` | `$HOME/vllm-env` | другое имя окружения |
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | другая модель |
| `MAX_MODEL_LEN` | `4096` | уменьшить на слабой карте |
| `GPU_UTIL` | `0.75` | на 6 ГБ `0.9` (как в задании) может не хватить — тогда `0.8` или дефолт |
| `PORT` | `8000` | порт занят |
| `EAGER` | `0` | `1` — без CUDA graphs (медленнее, но работает всегда) |

Примеры:

```bash
GPU_UTIL=0.8 bash scripts/start_vllm.sh          # типичный запуск на 6 ГБ
MAX_MODEL_LEN=2048 GPU_UTIL=0.8 bash scripts/start_vllm.sh
EAGER=1 bash scripts/start_vllm.sh               # если CUDA graphs падают
PORT=8001 bash scripts/start_vllm.sh
```

### Вариант Б — в фоне, с логом (окно можно закрыть)

```bash
cd /mnt/c/PythonProg/batch-generation-service
nohup bash scripts/start_vllm.sh > ~/vllm.log 2>&1 < /dev/null &
tail -f ~/vllm.log
```

`nohup` + перенаправление ввода/вывода нужно, чтобы сервер пережил закрытие терминала и `wsl.exe`. Проверено: после выхода из сеанса процесс продолжает работать.

> Не сохраняйте PID через `echo $! > ~/vllm.pid` при такой форме запуска: `$!` — это PID подоболочки от `cd ... && ... &`, а не сервера (на стенде получилось 22094 против реального 22096). PID ищите командой `pgrep -af "vllm serve"`.

### Вариант В — вручную, без скрипта

```bash
source ~/vllm-env/bin/activate

# nvcc из pip-пакета нужен torch.compile для захвата CUDA graphs
NVCC_BIN="$(python -c 'import glob, os, nvidia; print(os.path.dirname(glob.glob(os.path.join(os.path.dirname(nvidia.__file__), "cu*", "bin", "nvcc"))[0]))')"
export PATH="$NVCC_BIN:$PATH"
export HF_HUB_OFFLINE=1

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

### Что происходит при старте и сколько ждать

| Этап | Наблюдение на стенде |
|---|---|
| Загрузка весов на GPU | `Model loading took 2.98 GiB memory and 36.3 seconds` |
| Захват CUDA graphs | 51 граф, ~7–9 графов/с |
| KV-cache | 41 360 токенов при `max-model-len 4096` и `GPU_UTIL=0.8`, «maximum concurrency 10.1x» |
| Готовность | `INFO: Application startup complete.` → `/health` отдаёт 200 |
| Время старта | **204 с** холодный (в `outputs/report.json`), **~124 с** при прогретом кэше (замер 2026-10-23) |

### Особенности 6 ГБ и CUDA graphs

- **`GPU_UTIL=0.9` может не стартовать на 6 ГБ**: vLLM требует 0.9 × 6144 MiB = 5.4 GiB, а свободно бывает меньше. Рабочее значение на стенде — `0.8`. При 0.8 vLLM 0.31 сам резервирует память под CUDA graphs и всё равно даёт ~41k токенов KV-cache.
- **`PermissionError: 'nvcc'`** при захвате графов — не поломка: `torch.compile` вызывает `nvcc`, а он лежит в `site-packages/nvidia/cu*/bin` и не попадает в `PATH`. Скрипт добавляет каталог; резервный вариант — `EAGER=1` (без графов, медленнее).
- **`HF_HUB_OFFLINE=1`** ускоряет и стабилизирует старт, но требует скачанной модели (раздел 5). Если модель удалили, а кэш оставили — запустите один раз с `HF_HUB_OFFLINE=0`.

## 7. Проверка, что сервис работает

```powershell
# 1. Сервер жив
curl.exe -s -o NUL -w "%{http_code}`n" http://127.0.0.1:8000/health     # 200

# 2. Модель зарегистрирована
curl.exe -s http://127.0.0.1:8000/v1/models
# {"object":"list","data":[{"id":"Qwen/Qwen2.5-1.5B-Instruct",...,"max_model_len":4096,...

# 3. Полный smoke-тест: список моделей + холодный и тёплый запрос
python scripts\smoke_test.py
# base_url: http://127.0.0.1:8000/v1
# models: ['Qwen/Qwen2.5-1.5B-Instruct']
# cold: 0.59 s, prompt_tokens=49, completion_tokens=10
# warm: 0.20 s, prompt_tokens=49, completion_tokens=10

# 4. Проверка на реальных данных: 3 карточки из benchmark.jsonl
python run_batch.py --rows 3 --output outputs/smoke.jsonl --summary outputs/smoke.summary.json
```

Внутри Ubuntu те же проверки:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
ss -ltn | grep 8000        # LISTEN 0 2048 0.0.0.0:8000
```

Сервер слушает `0.0.0.0:8000` внутри WSL, Windows видит его как `127.0.0.1:8000` — отдельная настройка проброса портов не нужна.

> **`503` сразу после старта — это нормально.** Порт уже слушается, но движок ещё грузит веса и собирает CUDA graphs, поэтому `/v1/models` отвечает `503 Service Unavailable` (`openai.InternalServerError: Error code: 503`). `scripts/smoke_test.py` сам ждёт `/health` = 200 (до `--ready-timeout` секунд, по умолчанию 600) и печатает `ждём сервер: HTTP 503; повтор через 5 с`. Если ждать не нужно — `python scripts\smoke_test.py --ready-timeout 0`; если ожидание исчерпалось — смотреть лог vLLM и `pgrep -af "vllm serve"` в WSL.

## 8. Остановка сервиса

| Ситуация | Команда | Комментарий |
|---|---|---|
| Сервер на переднем плане | `Ctrl+C` в окне Ubuntu | штатное завершение |
| Сервер в фоне | `pkill -f "vllm serve"` | SIGTERM всем процессам vLLM; завершение за ~10 с |
| Нужен точный PID | `pgrep -af "vllm serve"` → `kill <PID>` | проверено: процесс исчезает, порт освобождается |
| Порт занят, PID неизвестен | `ss -ltnp \| grep 8000`, затем `kill <PID>` | PID — в последнем столбце |
| Не реагирует на SIGTERM | `pkill -9 -f "vllm serve"` | жёстко; GPU-память освободится |
| Остановить всё в WSL | `wsl --shutdown` (в PowerShell) | гасит **все** дистрибутивы, включая `docker-desktop`, если он установлен |
| Только один дистрибутив | `wsl --terminate Ubuntu-22.04` | эквивалент «выключения машины» Ubuntu |

Проверка, что сервис действительно остановлен:

```bash
pgrep -af "vllm serve" || echo "процессов нет"
ss -ltn | grep 8000 || echo "порт свободен"
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # память освободилась
```

```powershell
curl.exe -s -o NUL -w "%{http_code}`n" --max-time 3 http://127.0.0.1:8000/health   # 000 = не отвечает
```

Важно: `pkill -f "vllm serve"` ищет по всей командной строке, поэтому запускайте его **из обычного терминала Ubuntu**. Если обернуть его в `bash -lc "..."`, обёртка сама содержит эту строку и попадёт под шаблон.

Автозапуска у сервера нет: после перезагрузки Windows vLLM не поднимается сам, память GPU освобождена.

## 9. Полное удаление

Порядок: **сначала остановить сервис** (раздел 8), потом удалять данные, потом (при необходимости) дистрибутив.

### 9.1 Сколько места занимает что

| Что | Где | Размер на стенде |
|---|---|---|
| Окружение vLLM (torch + CUDA-колёса + flashinfer) | `~/vllm-env` | 9.9 ГБ |
| Кэш модели | `~/.cache/huggingface` | 2.9 ГБ |
| Кэш pip (все скачанные колёса) | `~/.cache/pip` | 5.2 ГБ |
| Кэш компиляции vLLM | `~/.cache/vllm` | 52 МБ |
| Кэш Triton / CUDA | `~/.triton`, `~/.nv` | ~5 МБ |
| Лог фонового запуска | `~/vllm.log` | единицы МБ |
| **Итого в WSL** | | **~18 ГБ** |
| Клиентское окружение проекта (Windows) | `<проект>\.venv` | по факту проекта |
| Остатки Linux-окружения в проекте | `<проект>\.venv-wsl` | 4.8 МБ |

### 9.2 Удалить только сервер, оставив Ubuntu

В терминале Ubuntu:

```bash
# 1. Убедиться, что ничего не работает
pgrep -af "vllm serve" || echo "процессов нет"
ss -ltn | grep 8000 || echo "порт свободен"

# 2. Окружение vLLM
rm -rf ~/vllm-env

# 3. Веса модели (2.9 ГБ)
rm -rf ~/.cache/huggingface
#    или только эта модель:
# rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct

# 4. Кэши установки и компиляции (5.2 ГБ + 52 МБ)
rm -rf ~/.cache/pip ~/.cache/vllm ~/.triton ~/.nv

# 5. Логи
rm -f ~/vllm.log

# 6. Проверка
df -h /home
```

Пакеты `python3-venv`/`python3-pip` можно оставить — они системные и занимают мегабайты.

### 9.3 Удалить клиентское окружение проекта (Windows, необязательно)

```powershell
cd C:\PythonProg\batch-generation-service
Remove-Item -Recurse -Force .venv        # клиентское окружение, если создавалось в проекте
Remove-Item -Recurse -Force .venv-wsl    # linux-окружение, оставшееся от экспериментов
```

Если каталога нет, PowerShell сообщит об ошибке — это нормально; чтобы не сообщал, добавьте `-ErrorAction SilentlyContinue`.

Код, `benchmark.jsonl`, `outputs/` при этом не затрагиваются.

### 9.4 Удалить дистрибутив Ubuntu целиком

Когда нужно освободить всё сразу (виртуальный диск WSL, домашний каталог, кэши):

```powershell
wsl --terminate Ubuntu-22.04
wsl --unregister Ubuntu-22.04
wsl -l -v
```

> ⚠️ `wsl --unregister` **необратимо уничтожает всё содержимое дистрибутива**: `/home`, `~/vllm-env`, кэш модели, логи. Файлы проекта на `C:` (смонтированные как `/mnt/c/...`) не удаляются. Виртуальный диск `ext4.vhdx` также будет удалён — место вернётся в Windows.

### 9.5 Удалить WSL полностью

Если WSL больше не нужен:

```powershell
wsl --shutdown

# удалить приложение WSL и его обновления (Windows 11 / свежая Windows 10)
wsl --uninstall

# отключить компоненты Windows (нужны права администратора)
dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /disable-feature /featurename:VirtualMachinePlatform /all /norestart

Restart-Computer
```

После перезагрузки `wsl --status` сообщит, что WSL не установлен. Дистрибутив, установленный из Microsoft Store («Ubuntu»), при необходимости удаляется как обычное приложение: Параметры → Приложения → Ubuntu → Удалить.

### 9.6 Что удалять не нужно

- **Драйвер NVIDIA в Windows** — он общий для системы и игр, к vLLM отношения не имеет; удаление сломает работу GPU.
- **Проект и артефакты** (`C:\PythonProg\batch-generation-service`) — это данные задания, не часть сервера.
- **GPU-память** освобождается сразу после остановки процесса, «чистить» её вручную не требуется.

## 10. Типичные проблемы

| Симптом | Причина | Решение |
|---|---|---|
| `bash: /mnt/c/.../start_vllm.sh: Permission denied` | каталог `/mnt/c` смонтирован без `chmod`, файл не исполняемый | запускать `bash scripts/start_vllm.sh`, а не `./scripts/start_vllm.sh` |
| `Не найдено окружение /home/user/vllm-env` | окружение не создано или создано в другом месте | `python3 -m venv ~/vllm-env` (раздел 4) или `VENV=/путь bash scripts/start_vllm.sh` |
| `PermissionError: [Errno 13] Permission denied: 'nvcc'` при старте | `torch.compile` не находит `nvcc` из pip-пакета | скрипт добавляет `PATH` сам; резерв — `EAGER=1 bash scripts/start_vllm.sh` |
| `ValueError: Free memory ... is less than desired GPU memory utilization` / OOM | `GPU_UTIL=0.9` на 6 ГБ не помещается | `GPU_UTIL=0.8`, при необходимости `MAX_MODEL_LEN=2048` |
| Сервер падает с `OSError: We couldn't connect to 'https://huggingface.co'` | `HF_HUB_OFFLINE=1`, а модели нет в кэше | `HF_HUB_OFFLINE=0` на первый запуск или `hf download Qwen/Qwen2.5-1.5B-Instruct` |
| `Address already in use` / порт 8000 занят | висит старый процесс vLLM | `pkill -f "vllm serve"`; либо `PORT=8001` |
| `nvidia-smi: command not found` в WSL | старый WSL/ядро, WSL1, драйвер Windows | `wsl --update`, `wsl --set-default-version 2`, обновить драйвер NVIDIA |
| `wsl: your ... screen size is bogus` при вызове из PowerShell | безобидное предупреждение `wsl.exe` при нестандартном размере окна | игнорировать |
| Генерация резко замедлилась | тепловой троттлинг карты | `nvidia-smi -q -d PERFORMANCE` (ищите `SW Thermal Slowdown`), дать карте остыть |
| Клиент получает `Connection refused` | сервер не запущен или слушает другой порт | проверить `/health`, `ss -ltn \| grep 8000`, совпадение `PORT` и `--base-url` |

## 11. Чек-лист «с нуля» (копипаст)

```powershell
# --- Windows, PowerShell от администратора ---
wsl --update
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
wsl -l -v
```

```bash
# --- Ubuntu: окружение и vLLM ---
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
python -m pip install -U pip
pip install vllm==0.31.0
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
nvidia-smi

# --- модель ---
hf download Qwen/Qwen2.5-1.5B-Instruct

# --- запуск (на 6 ГБ) ---
cd /mnt/c/PythonProg/batch-generation-service
GPU_UTIL=0.8 bash scripts/start_vllm.sh
```

```powershell
# --- проверка из Windows ---
curl.exe -s -o NUL -w "%{http_code}`n" http://127.0.0.1:8000/health
python scripts\smoke_test.py
```

```bash
# --- остановка ---
pkill -f "vllm serve"      # в терминале Ubuntu; или Ctrl+C в окне сервера
```

```bash
# --- удаление (в терминале Ubuntu) ---
rm -rf ~/vllm-env ~/.cache/huggingface ~/.cache/pip ~/.cache/vllm ~/.triton ~/.nv ~/vllm.log
```

```powershell
# --- удаление дистрибутива целиком (необратимо) ---
wsl --terminate Ubuntu-22.04
wsl --unregister Ubuntu-22.04
```

## Связанные документы

- `README.md` — результаты замеров, конфигурация генерации, стоимость.
- `Plan.md` — журнал решений и чек-лист приёмки.
- `scripts/start_vllm.sh` — скрипт запуска с параметрами по умолчанию.




