# AI Inference RPC Gateway

![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.10%2B-064F8C?style=flat-square&logo=cmake&logoColor=white)
![Muduo](https://img.shields.io/badge/Muduo-TCP%20Reactor-0A7EA4?style=flat-square)
![ONNX%20Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20Only-005CED?style=flat-square)
![Protobuf](https://img.shields.io/badge/Protobuf-RPC%20Envelope-3367D6?style=flat-square)
![ZooKeeper](https://img.shields.io/badge/ZooKeeper-Service%20Discovery-D22128?style=flat-square)

> 这是为了研究“高并发网络 I/O 如何与底层模型推理衔接”而做的个人项目。实现了一条完整的推理链路：请求接入、动态批处理、CPU 推理、以及可选的服务注册发现。

---

## 项目简介

项目整体实现采用 **Muduo + Protobuf** 处理 TCP RPC 接入，通过 `BatchingQueue` 在服务端内部做动态批处理，再把聚合后的批次交给 **ONNX Runtime（CPU Execution Provider）** 执行推理。同时，项目补充了一个最小可用的 **ZooKeeper 注册发现** 流程，用来验证网关在接近真实部署形态时的运行方式。

核心路径可以概括为：

- `RpcServer`：接收连接、完成协议解码与响应发送
- `InferenceServiceImpl`：负责请求校验、超时控制与错误响应构造
- `BatchingQueue`：负责请求聚合、批次触发与结果分发
- `InferenceEngine`：负责图像预处理、张量构造与 ONNX Runtime 推理
- `ZkClient` / `ServiceRegistry`：负责 ZooKeeper 连接与服务注册

---

## 核心特性

### 1. 动态批处理（Dynamic Batching）

外部请求始终以“单张图片、单次 RPC”的形式进入，但服务端内部不会逐条立即执行推理，而是把短时间窗口内的请求聚合起来，并在以下任一条件满足时触发批推理：

- 达到最大批大小 `max_batch_size`
- 达到最大等待时延 `max_latency_ms`

实现上使用了 C++ 标准并发原语：

- `std::mutex`
- `std::condition_variable`
- `std::promise / std::future`
- 后台工作线程

这套机制的重点在于：

1. **请求聚合对调用方透明**，上层不需要感知服务端是否发生了批处理。
2. **结果分发是精确的**，每个请求都通过自己的 `future` 返回结果或异常。

此外，`InferenceServiceImpl` 不会在 Muduo 的 I/O 线程中直接等待推理结果，而是把等待 `future` 的动作转移到业务线程池，避免网络线程被阻塞。

---

### 2. 轻量预处理：`stb_image` / `stb_image_resize2`

项目没有引入 OpenCV，而是采用更轻量的预处理链路：

- `stb_image.h` 负责图像解码
- `stb_image_resize2.h` 负责图像缩放
- 后续手动完成归一化和 `NCHW` 张量布局转换

这样做的主要考虑：

- 减少依赖复杂度
- 保持预处理路径清晰可控
- 更适合这个项目以“推理链路研究”为主的定位

当前 `InferenceEngine::Preprocess(...)` 会将图像缩放到 `224x224`，再按 ImageNet 的均值与方差完成归一化，并转换成模型输入张量。

这一步的重点是**预处理行为的稳定性与可复现性**。在跨语言部署场景下，真正影响结果一致性的通常不是模型本身，而是：

- resize 行为
- 像素采样方式
- `HWC / NCHW` 布局变换
- normalization 流程

---

### 3. Muduo + ZooKeeper 的最小服务治理链路

项目补充了一个最小可用的注册发现流程，用来验证网关在接近真实部署形态时的运行方式。

当提供 ZooKeeper 地址时，网关会在启动后尝试注册服务实例，路径约定如下：

```text
/ai-rpc-gateway/services/<service_name>/providers/<ip:port>
```

这里的设计点：

- 实例节点使用 **ephemeral node（临时节点）**
- ZooKeeper 失败时，**主 RPC 服务仍继续工作**

也就是说，ZooKeeper 在这里是增强能力，而不是主推理链路的强依赖。

---

## 核心模块概览

| 模块 | 位置 | 作用 |
| --- | --- | --- |
| `InferenceEngine` | `include/inference_engine.h` / `src/inference_engine.cpp` | 模型加载、图像预处理、单样本/批量推理 |
| `BatchingQueue` | `include/batching_queue.h` / `src/batching_queue.cpp` | 请求聚合、批次触发、future 结果分发、队列背压 |
| `InferenceServiceImpl` | `include/inference_service_impl.h` / `src/inference_service_impl.cpp` | RPC 请求校验、超时控制、异步等待、错误响应构造 |
| `RpcServer` | `include/rpc_server.h` / `src/rpc_server.cpp` | Muduo TCP 监听、协议解码、连接管理、响应回写 |
| `ZkClient` | `include/zk_client.h` / `src/zk_client.cpp` | ZooKeeper 连接管理、持久路径创建、临时节点创建 |
| `ServiceRegistry` | `include/service_registry.h` / `src/service_registry.cpp` | 服务注册路径约定与实例注册 |

---

## 性能基准测试（Benchmarks）

仓库里保留了真实压测结果，见 `stress_test_report.md`。

### 测试条件

- 环境：**纯 CPU，Mac M 系列 / OrbStack Ubuntu ARM64**
- 目标服务：`127.0.0.1:50051`
- 压测脚本：`scripts/stress_test.py`
- 并发档位：`50 / 200 / 500 / 1000`
- 每档请求数：`4000`
- 单请求超时：`10000 ms`

### 实测结果

| 并发数 | 总请求数 | 成功数 | 失败数 | 成功率 | QPS | P99 延迟 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 4000 | 4000 | 0 | 100.00% | 23.95 | 2286.44 ms |
| 200 | 4000 | 4000 | 0 | 100.00% | 23.84 | 8529.65 ms |
| 500 | 4000 | 235 | 3765 | 5.88% | 2.93 | 9844.00 ms |
| 1000 | 4000 | 0 | 4000 | 0.00% | 0.00 | N/A |

### 我对这组数据的理解

这组结果很符合一个纯 CPU 服务在接近容量上限时的表现：

- 并发从 `50` 提到 `200`，**QPS 基本不再提升**，说明系统已经接近稳定吞吐上限
- 但这时 **P99 延迟继续大幅变差**，说明队列等待时间正在累积
- 到 `500` 并发时，大量请求已经在超时前等不到结果
- 到 `1000` 并发时，系统已经完全进入过载区

如果用排队论的语言说，就是：当请求到达率 `λ` 逼近服务能力 `μ` 时，系统不一定立刻掉吞吐，但尾延迟会先失控；再往上推，超时和失败才会集中出现。

可以得出结论：

- 当前稳定极限 QPS 大约在 **24 左右**
- **并发 50 左右** 是更合理的工作点
- 再继续加压，主要增加的是排队时间，而不是吞吐收益

> 对在线推理服务来说，吞吐上限往往不是最先暴露问题的指标，P99 才是。

---

## 快速开始（Quick Start）

### 1. 前置依赖

请先准备以下依赖：

- CMake `>= 3.10`
- 支持 C++17 的编译器
- Protobuf
- Boost
- Zlib
- Muduo 相关库
  - `muduo_base`
  - `muduo_net`
  - `muduo_protobuf_codec`
  - `muduo_protorpc`
  - `muduo_protorpc_wire`
- ONNX Runtime（**CPU 版本**）
- ZooKeeper C Client（`zookeeper_mt`，可选）

> 本项目明确只考虑 **CPU 推理**，不包含 CUDA / TensorRT 路径。

### 2. 构建项目

```bash
cmake -B build
cmake --build build -j
```

构建完成后，主要可执行文件包括：

- `build/gateway`
- `build/test_inference`
- `build/test_batching`
- `build/test_rpc_client`

项目当前也会导出 clangd 所需的编译数据库：

```text
build/compile_commands.json
```

根目录软链接：

```text
compile_commands.json -> build/compile_commands.json
```

### 3. 启动网关服务

#### 不接 ZooKeeper

```bash
./build/gateway ./models/resnet18.onnx 127.0.0.1 50051
```

#### 接入 ZooKeeper

```bash
./build/gateway ./models/resnet18.onnx 127.0.0.1 50051 127.0.0.1:2181
```

### 4. 运行 Python 压测脚本

#### 单轮压测

```bash
python3 ./scripts/stress_test.py \
  --host 127.0.0.1 \
  --port 50051 \
  --requests 1000 \
  --concurrency 50 \
  --timeout-ms 5000
```

#### 阶梯式压测

```bash
python3 ./scripts/stress_test.py \
  --host 127.0.0.1 \
  --port 50051 \
  --requests 4000 \
  --concurrency-levels 50,200,500,1000 \
  --timeout-ms 10000
```

脚本会输出：

- 总请求数
- 成功 / 失败数
- QPS
- P50 / P95 / P99 延迟
- 平均延迟

---

## 仓库目录结构

```text
.
├── CMakeLists.txt
├── CLAUDE.md
├── project_plan.md
├── prompt_readme.txt
├── compile_commands.json -> build/compile_commands.json
├── include
│   ├── batching_queue.h
│   ├── inference_engine.h
│   ├── inference_service_impl.h
│   ├── rpc_protocol.h
│   ├── rpc_server.h
│   ├── service_registry.h
│   └── zk_client.h
├── src
│   ├── batching_queue.cpp
│   ├── inference_engine.cpp
│   ├── inference_service_impl.cpp
│   ├── main.cpp
│   ├── rpc_protocol.cpp
│   ├── rpc_server.cpp
│   ├── service_registry.cpp
│   └── zk_client.cpp
├── protos
│   └── inference.proto
├── models
│   ├── resnet18.onnx
│   └── resnet18.onnx.data
├── scripts
│   ├── export_model.py
│   └── stress_test.py
├── tests
│   ├── test_batching.cpp
│   ├── test_inference.cpp
│   └── test_rpc_client.cpp
├── third_party
│   ├── stb_image.h
│   └── stb_image_resize2.h
└── build
```

---

## 我做这个项目时比较关心的点

- C++ 服务端里的 dynamic batching 怎么实现
- Muduo 和 ONNX Runtime 怎么衔接
- 轻量预处理链路怎么自己控制
- 简单的 ZooKeeper 注册发现怎么接进来

所以如果你也在关心类似的问题，那么这个仓库应该会有一些参考价值。
