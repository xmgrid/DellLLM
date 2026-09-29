#!/bin/bash
# 头节点 GID 修好之后，按「先停两边、再先工作节点、后头节点」重启推理。
# 光纤中断后 systemd 仍可能显示服务在跑，单靠 Restart=on-failure 不会重来。
set -uo pipefail
exec 9>/run/lock/spark-recover.lock
flock -n 9 || exit 0

gid_ok() {
  g1=$(cat /sys/class/infiniband/rocep1s0f1/ports/1/gids/3 2>/dev/null || true)
  g2=$(cat /sys/class/infiniband/roceP2p1s0f1/ports/1/gids/3 2>/dev/null || true)
  case "$g1" in *c80c*) ;; *) return 1 ;; esac
  case "$g2" in *c910*) ;; *) return 1 ;; esac
}

for _ in $(seq 1 20); do
  gid_ok && break
  sleep 2
done
if ! gid_ok; then
  logger -t spark-recover "head GID3 still wrong, leaving the engine stopped"
  exit 0
fi

logger -t spark-recover "GID3 restored, restarting the cluster"
systemctl stop spark-head || true

worker_ssh() {
  sudo -u linhaixiang -H ssh -o BatchMode=yes -o ConnectTimeout=8 \
    linhaixiang@192.168.0.79 "$@"
}

stopped=0
for _ in $(seq 1 24); do
  if worker_ssh sudo -n systemctl stop spark-worker; then
    stopped=1
    break
  fi
  sleep 5
done
if [ "$stopped" != 1 ]; then
  logger -t spark-recover "worker did not accept stop, not starting the head"
  exit 0
fi

worker_ssh sudo -n systemctl reset-failed spark-worker || true
worker_ssh sudo -n systemctl start spark-worker
sleep 3
systemctl reset-failed spark-head || true
systemctl start spark-head
logger -t spark-recover "cluster start issued"
