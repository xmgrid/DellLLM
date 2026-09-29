# 双机 GB10 上的 DeepSeek-V4-Flash-0731

两台 NVIDIA GB10 用两条 200Gb RoCE 光纤组成 TP=2，原生进程加载本地检查点，不使用容器。控制台在头节点的 9090 端口，OpenAI 兼容接口在 8888 端口。

这是这套机器当前正在使用的配置和自有源码。网卡地址、主机名按这套部署写死。换机器时要改启动脚本、GID 修复脚本和控制台里的地址。仓库里没有登录密码。

## 机器

| 角色 | 主机名 | 管理网 | 光纤 1 | 光纤 2 | RoCE GID 第 3 项 |
|---|---|---|---|---|---|
| 头节点 | promaxgb10-84ce | 192.168.0.78 | enp1s0f1np1 `192.168.200.12` MAC `fc:4c:ea:f9:84:cf` | enP2p1s0f1np1 `192.168.201.16` MAC `fc:4c:ea:f9:84:d3` | `c80c` / `c910` |
| 工作节点 | spark-8505 | 192.168.0.79 | enp1s0f1np1 `192.168.200.13` MAC `fc:4c:ea:f9:85:06` | enP2p1s0f1np1 `192.168.201.17` MAC `fc:4c:ea:f9:85:0a` | `c80d` / `c911` |

两条光纤分属 `192.168.200.0/24` 和 `192.168.201.0/24`，不绑定、不放进同一个网段。地址用 netplan 的 `match.macaddress` 钉在网卡上，链路抖动后不会互换。

内核 `7.0.0-1019-nvidia`，驱动 `580.178.04`，使用发行版签名的 `linux-modules-nvidia-580-open`，不要用未登记到安全启动的 DKMS 模块。启动参数需要 `kho=off`（软件包 `nvidia-spark-grub-kho`）。内核 7.0 默认打开的 KHO 会在显存占满时让 `ibv_reg_mr` 返回内存不足。`nvidia-drm` 的 modeset 设为 1，以便 DMA-BUF。

检查点目录：`/home/linhaixiang/model/DeepSeek-V4-Flash-0731`。运行时是解包到 `/opt/dspark` 的 vLLM，用 `torch.distributed.run` 启动，不是 `docker run`。

## 推理参数

头节点脚本 `sparkctl/dspark-head.sh`，工作节点 `sparkctl/dspark-worker.sh`。两边环境一致：

- `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`
- `NCCL_SOCKET_IFNAME=enp1s0f1np1`，`NCCL_IB_GID_INDEX=3`
- `NCCL_DMABUF_ENABLE=1`，`NCCL_NET_GDR_LEVEL=0`，`NCCL_CUMEM_ENABLE=0`
- `NCCL_IB_TIMEOUT=22`，`NCCL_IB_RETRY_CNT=7`
- 上下文 `1048576`，`gpu-memory-utilization 0.835`，KV 为 `nvfp4_ds_mla`
- `max-num-seqs 6`，`max-num-batched-tokens 8192`
- CUDA graph：`FULL_DECODE_ONLY`，捕获大小 `1,2,4,8,16,32`
- MoE：`flashinfer_b12x`。不要把 linear backend 也设成 `flashinfer_b12x`
- 推测解码：`dspark`，`num_speculative_tokens` 为 **5**，`draft_sample_method=greedy`。官方说明里的 7 与这套速度配置不同，界面按 5 显示

启动前脚本会检查本机光纤地址和 GID。GID 第 3 项是全 0 时拒绝启动。

## 服务

| 单元 | 节点 | 作用 |
|---|---|---|
| `spark-head.service` | 头节点 | 推理 rank 0，开机自启 |
| `spark-worker.service` | 工作节点 | 推理 rank 1，开机自启 |
| `sparkctl.service` | 头节点 | 控制台 9090 |
| `spark-recover.service` | 头节点 | 一次性。GID 修好后先停两边，再先启动工作节点、后启动头节点。不要 enable |
| `99-spark-roce-gid` | 两边 | NetworkManager dispatcher。链路 up 后若 GID 第 3 项丢了，只抖动对应的那条连接 |
| `99-spark-rpfilter.conf` | 两边 | 光纤接口的 `rp_filter=2` |

空闲时头节点每 5 秒向工作节点广播一次空批次。否则工作节点会堵在集合通信里，大约 30 分钟后超时退出，而 systemd 仍显示服务在运行。这段修改在两边的 `vllm/v1/engine/core.py`：`_mirror_spmd_inputs` 里，rank 0 无任务时对 `input_queue.get` 使用 `timeout=5`。

## 控制台

`sparkctl/server.py` 提供页面和 `/api/chat`。纯文本走 `/v1/completions`，图片走 `/v1/chat/completions`。生成长度默认可以填到 1048576，服务端会先扣掉提示词和图片占用的位置。语言选择只加在发往模型的最后一条用户消息上。

`encoding_dsv4.py` 是检查点附带的官方编码，和 `server.py` 放在同一目录。它不属于这份自有源码，这里不转载。

视觉实现在 `sparkctl/vision_v4.py`，由 `patch_runtime.py` 装进 `/opt/dspark` 下的模型目录。图像 token 使用词表内的 `<｜image｜>`。JPG/PNG 按原始字节上传，不再重新压缩。

## 启动顺序

先确认四条光纤的地址和 GID，再启动工作节点，然后启动头节点。权重加载大约需要几分钟。不要在前一次 GPU 进程还没退出时重叠启动。
