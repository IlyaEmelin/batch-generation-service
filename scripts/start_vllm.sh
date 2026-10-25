#!/usr/bin/env bash
# Запуск vLLM-сервера в WSL2.
#
# Нативный Windows не поддерживается, поэтому сервер живёт в Ubuntu (WSL2),
# а проект может оставаться на /mnt/c. Окружение vLLM создаётся отдельно от
# проекта: ~/vllm-env (см. README, раздел «Как воспроизвести»).
#
# Значения по умолчанию соответствуют заданию. На 6 ГБ памяти RTX 2060
# параметры задания не стартуют: vLLM требует 0.9 * 6.0 = 5.4 GiB, а свободно
# бывает меньше. Поэтому значения можно переопределить переменными окружения:
#
#   MAX_MODEL_LEN=2048 GPU_UTIL=0.8 bash scripts/start_vllm.sh
#
# Каталог /mnt/c смонтирован без прав на chmod, поэтому скрипт запускается
# через `bash`, а не как исполняемый файл.
#
# Скрипт печатает время старта, чтобы его можно было сверить с отчётом.
#
# EAGER=1 включает --enforce-eager: без CUDA graphs меньше пиковая память, но
# ниже скорость. По умолчанию графы включены, потому что 6 ГБ хватает
# (KV-cache занимает единицы процентов), а замер на concurrency 4 показал
# просадку без них.
set -euo pipefail

VENV="${VENV:-$HOME/vllm-env}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.75}"
PORT="${PORT:-8000}"
EAGER="${EAGER:-0}"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Не найдено окружение $VENV. Создайте его: python3 -m venv ~/vllm-env" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# torch.compile (Inductor) вызывает nvcc при построении CUDA graphs. В pip-сборке
# torch тулчейн лежит в site-packages/nvidia/cu*/bin и в PATH не попадает, из-за
# чего захват графов падает с PermissionError: 'nvcc'. Добавляем каталог явно;
# при EAGER=1 он не нужен, но безвреден.
NVCC_PATH="$(python -c 'import glob, os, nvidia; print((glob.glob(os.path.join(os.path.dirname(nvidia.__file__), "cu*", "bin", "nvcc")) or [""])[0])' 2>/dev/null || true)"
if [[ -n "$NVCC_PATH" ]]; then
  export PATH="$(dirname "$NVCC_PATH"):$PATH"
fi

# Загрузка весов не должна попадать в замер времени старта: если модель уже
# скачана, работаем из кэша без обращений к HF Hub. Если модель ещё не
# скачана, сначала выполните: hf download Qwen/Qwen2.5-1.5B-Instruct
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "start_vllm: model=$MODEL max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_UTIL port=$PORT offline=$HF_HUB_OFFLINE eager=$EAGER" >&2
echo "start_vllm: старт $(date -Is)" >&2

EXTRA_ARGS=()
if [[ "$EAGER" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

exec vllm serve "$MODEL" \
  --dtype half \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"

