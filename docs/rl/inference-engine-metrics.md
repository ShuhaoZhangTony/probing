# 推理引擎指标展示（SGLang 与 vllm-hust）

Probing 定期从推理引擎抓取 Prometheus 指标，归一化后在 Web UI 的 **`/inference`** 页面展示。当前支持 **SGLang** 和 GenNova 使用的 **vllm-hust**；两者均不要求 Slime。

## 功能概述

Probing 后台每隔约 5 秒从已注册的推理引擎 **指标端点** 抓取一次 Prometheus 文本，将各引擎不同的指标名映射为统一的 **7 个归一化指标**，供页面上的数字卡片与折线图使用。历史采样写入 SQL 表 `python.inference_engine_metric`（`metric_name` 以 `normalized.` 为前缀）。

指标路径可配置：

| 部署方式 | 默认路径 |
|----------|----------|
| Slime `sglang_router` / model gateway | `/engine_metrics` |
| `sglang.launch_server --enable-metrics` | `/metrics` |
| `vllm-hust` | `/metrics` |

注册引擎时通过 `metrics_path` 指定，或设置环境变量 `PROBING_ENGINE_METRICS_PATH`。

## 数据流

```
推理引擎指标端点（/metrics 或 /engine_metrics）
  → Probing 后台定期抓取（默认每 5 秒）
  → 归一化为 7 个 canonical 指标
  → 写入 python.inference_engine_metric，并更新引擎 registry 中的最新值
  → Web UI /inference 页面轮询展示（卡片 + 折线图）
```

## Web UI 布局

`/inference` 页面包含三个区块：

![Inference 页面：已注册引擎列表与最新归一化指标卡片](./assets/placeholders/engine_summary.png)

![Inference 页面：7 项指标的时间序列折线图](./assets/placeholders/engine_metric.png)

1. **Registered engines** — 已注册引擎列表（引擎 ID、类型、指标 URL、健康状态）
2. **Latest normalized metrics** — 各引擎 7 项指标的最新数值卡片
3. **Metric trends (time series)** — 7 项指标随时间变化的折线图（默认每 5 秒自动刷新，也可点击 **Scrape now** 立即抓取）

## 归一化指标

页面固定展示以下 7 项指标（内部 key 用于存储与 API，UI 为可读标题）：

| UI 名称 | 内部 key | 含义 |
|---------|----------|------|
| In-flight requests | `inflight_requests` | 当前正在执行、尚未完成的推理请求数量 |
| Queue depth | `queue_depth` | 已进入系统但尚在排队、等待调度的请求数量 |
| Throughput (tok/s) | `throughput_tps` |  token 生成吞吐量，单位 tokens/s |
| TPOT (ms) | `tpot_ms` | Time Per Output Token：每生成一个输出 token 的平均耗时（毫秒） |
| TTFT (ms) | `ttft_ms` | Time To First Token：从请求到达到产出第一个 token 的平均耗时（毫秒） |
| KV cache usage | `kv_cache_usage_ratio` | KV cache 内存池占用比例，取值 0～1 |
| Cache hit ratio | `cache_hit_ratio` | Prefix cache（radix cache）命中率，取值 0～1 |

若某时刻引擎未暴露对应原始指标，或尚无推理负载，卡片可能显示 `—`，折线图可能暂无采样点。

### vllm-hust 的窗口语义

`vllm-hust` 的并发数、队列深度和 KV cache 占用是即时 gauge；吞吐、
TTFT、TPOT 和 prefix-cache 命中率来自计数器或直方图，必须由连续两次
抓取的差分计算：

| 归一化指标 | vllm-hust 原始指标/计算 |
|------------|-------------------------|
| `inflight_requests` | 所有实例的 `vllm:num_requests_running` 之和 |
| `queue_depth` | 所有实例的 `vllm:num_requests_waiting` 之和 |
| `kv_cache_usage_ratio` | `vllm:kv_cache_usage_perc` 的实例最大值 |
| `throughput_tps` | `delta(vllm:generation_tokens_total) / delta(time)` |
| `ttft_ms` | `delta(vllm:time_to_first_token_seconds_sum) / delta(..._count) * 1000` |
| `tpot_ms` | `delta(vllm:request_time_per_output_token_seconds_sum) / delta(..._count) * 1000` |
| `cache_hit_ratio` | `delta(vllm:prefix_cache_hits_total) / delta(vllm:prefix_cache_queries_total)` |

首次抓取只发布三个 gauge。若计数器重启、标签集合变化或窗口内没有可计算
的分母，相应派生指标在该窗口缺失，而不是伪造为 0。原始样本仍写入
`python.inference_engine_metric`，便于排查。

## 如何使用

以下说明如何在 Probing 中接入并展示 **真实 SGLang 推理服务**（`sglang.launch_server`）。

### vllm-hust / GenNova 注册

先确认 vllm-hust 的 Prometheus 端点可访问，再向 Probing 注册。模型和运行时
修订是必填 provenance，避免 GenNova 消费来自错误模型或二进制的快照：

```bash
curl -G 'http://127.0.0.1:8080/apis/pythonext/engines/register' \
  --data-urlencode 'router_addr=http://127.0.0.1:8000' \
  --data-urlencode 'engine_id=qwen35-prod' \
  --data-urlencode 'engine_type=vllm-hust' \
  --data-urlencode 'model_id=Qwen/Qwen3.5-35B-A3B' \
  --data-urlencode 'runtime_revision=d0f22d2bda562156e4dbf433ce645e1769b4f804' \
  --data-urlencode 'accelerator_backend=vllm-ascend-hust'
```

也可在同一 Python 进程中调用 `register_vllm_hust(...)`。注册入口固定抓取
`/metrics`，并保留 `model_id`、`runtime_revision`、
`accelerator_backend` 和 `integration=gennova` 标签。

### 前置：构建 Web UI

在仓库根目录执行（需已安装 [Dioxus CLI](https://dioxuslabs.com/learn/0.6/getting_started/) `dx`）：

```bash
make frontend
```

完成后 `web/dist` 会链接到嵌入式 Web 产物目录；如需显式磁盘覆盖，可将 `PROBING_ASSETS_ROOT` 设为 `web/dist`。

### 步骤 1：启动 SGLang 推理服务

在 GPU 节点启动 SGLang，**必须**加上 `--enable-metrics`，指标暴露在 `/metrics`：

```bash
python -m sglang.launch_server \
  --model-path <你的模型路径> \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics
```

确认指标端点可用：

```bash
curl -s http://127.0.0.1:30000/metrics | head
```

### 步骤 2：启动 Probing 并注册引擎

在仓库根目录、另一终端执行：

```bash
export PROBING=1
export PROBING_PORT=8080
export PROBING_ASSETS_ROOT=web/dist

python examples/inference/sglang_inference_metrics_demo.py \
  --engine-url http://127.0.0.1:30000 \
  --engine-id sglang-real \
  --num-requests 0 \
  --keep-alive-seconds 3600
```

说明：

- `PROBING_PORT` **必须设置**，否则只有 Unix socket，浏览器无法访问
- `PROBING_ASSETS_ROOT` 指向已构建的前端目录 `web/dist`
- 默认注册 `http://127.0.0.1:30000`，指标路径为 `/metrics`（由 demo 对外部引擎自动选择）
- 进程保持运行，后台每约 5 秒抓取一次指标

可修改 `--engine-url`、`--engine-id` 或 `PROBING_PORT` 等参数。

### 步骤 3：打开 Web UI

浏览器访问：

```
http://127.0.0.1:8080/inference
```

若在远程机器上运行，需将 `8080` 端口转发到本机后再访问。

### 步骤 4（推荐）：发送推理请求以产生指标

空闲时许多指标为 0。可持续向 SGLang 发 `/generate` 请求，例如：

```bash
while true; do
  curl -s -X POST http://127.0.0.1:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"Explain RL in one paragraph.","max_new_tokens":64}' \
    > /dev/null
  sleep 0.5
done
```

发送负载后，**Throughput**、**TPOT**、**TTFT**、**Cache hit ratio** 等折线会出现非零变化；页面约每 5 秒自动刷新，也可点击 **Scrape now** 立即抓取。

### 常用 API（可选）

| 接口 | 作用 |
|------|------|
| `GET /apis/pythonext/engines/snapshot` | 查看已注册引擎及最新归一化指标 |
| `GET /apis/pythonext/engines/scrape` | 立即触发一次指标抓取 |
| `GET /apis/pythonext/engines/register?...` | 手动注册 SGLang 或 vllm-hust；后者要求 `model_id` 与 `runtime_revision` |

### 相关环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PROBING_PORT` | （未设置） | 设为 `8080` 等端口以启用 HTTP Web UI |
| `PROBING_ASSETS_ROOT` | （未设置） | Web 静态资源目录，通常为 `web/dist` |
| `PROBING_ENGINE_SCRAPE` | `1` | 是否启用后台定期抓取 |
| `PROBING_ENGINE_SCRAPE_INTERVAL` | `5` | 抓取间隔（秒） |
| `PROBING_ENGINE_METRICS_PATH` | `/engine_metrics` | 注册时未指定 `metrics_path` 的默认指标路径 |
