#!/bin/bash
# 头节点启动脚本。推理进程直接跑解包后的 /opt/dspark，不使用容器。
# 两条 200Gb 光纤分属 192.168.200.0/24 和 192.168.201.0/24，不做绑定。
# 地址由网卡 MAC 钉死；GID 第 3 项必须是 RoCEv2，对不上就拒绝启动。
set -euo pipefail
PY=/opt/dspark/usr/bin/python3.12
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=/opt/dspark/usr/local/lib/python3.12/dist-packages
export LD_LIBRARY_PATH=/opt/dspark/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/dspark/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_HUB_OFFLINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_HOME=/usr/local/cuda-13.0
export CPATH=/usr/local/cuda-13.0/targets/sbsa-linux/include${CPATH:+:$CPATH}
export PYTHONUNBUFFERED=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_HOST_IP=192.168.200.12
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_CROSS_NIC=0
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export NCCL_DEBUG=WARN
export NCCL_DMABUF_ENABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
# 光纤地址按 MAC 钉死。链路抖动后如果地址或 GID 对不上，直接退出。
if ! ip -4 -o addr show dev enp1s0f1np1 | grep -q '192.168.200.12/'; then
  echo "refusing to start: enp1s0f1np1 is not 192.168.200.12" >&2
  ip -4 -br addr >&2 || true
  exit 1
fi
if ! ip -4 -o addr show dev enP2p1s0f1np1 | grep -q '192.168.201.16/'; then
  echo "refusing to start: enP2p1s0f1np1 is not 192.168.201.16" >&2
  exit 1
fi
g1=$(cat /sys/class/infiniband/rocep1s0f1/ports/1/gids/3 2>/dev/null || true)
g2=$(cat /sys/class/infiniband/roceP2p1s0f1/ports/1/gids/3 2>/dev/null || true)
case "$g1" in *c80c) ;; *) echo "refusing to start: rocep1 GID3=$g1" >&2; exit 1 ;; esac
case "$g2" in *c910) ;; *) echo "refusing to start: roceP2 GID3=$g2" >&2; exit 1 ;; esac
exec "$PY" -m torch.distributed.run \
  --nnodes=2 --node-rank=0 \
  --master_addr=192.168.200.12 --master_port=29500 \
  --nproc_per_node=1 \
  -m vllm.entrypoints.openai.api_server \
  --model /home/linhaixiang/model/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend external_launcher \
  --trust-remote-code \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.835 \
  --kv-cache-dtype nvfp4_ds_mla \
  --tokenizer-mode deepseek_v4 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 8192 \
  --block-size 256 \
  --host 0.0.0.0 --port 8888 \
  --limit-mm-per-prompt '{"image":4}' \
  --chat-template /home/linhaixiang/sparkctl/chat_template.jinja \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --kernel-config '{"enable_cutedsl_warmup":true,"enable_flashinfer_autotune":false,"moe_backend":"flashinfer_b12x"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'
