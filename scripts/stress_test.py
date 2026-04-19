import argparse
import base64
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List


# 为了复用项目虚拟环境中的 protobuf 依赖，优先把 .venv 的 site-packages 注入当前进程。
def add_project_venv_site_packages(project_root: Path) -> None:
    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    result = subprocess.run(
        [str(venv_python), "-c", "import site; import sys; sys.stdout.write(chr(10).join(site.getsitepackages()))"],
        check=True,
        capture_output=True,
        text=True,
    )
    for site_path in reversed([line.strip() for line in result.stdout.splitlines() if line.strip()]):
        if site_path not in sys.path:
            sys.path.insert(0, site_path)


# 动态调用 protoc 生成 Python 版本的 protobuf 模块，避免手动维护产物文件。
def build_python_proto(project_root: Path) -> Path:
    proto_path = project_root / "protos" / "inference.proto"
    output_dir = Path(tempfile.mkdtemp(prefix="ai_rpc_gateway_proto_"))
    subprocess.run(
        [
            "protoc",
            f"--proto_path={project_root / 'protos'}",
            f"--python_out={output_dir}",
            str(proto_path),
        ],
        check=True,
    )
    return output_dir


# 生成稳定的内存内测试图像，避免随机字节导致服务端图像解码结果不可预期。
def generate_dummy_image(size: int) -> bytes:
    del size
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0JYAAAAASUVORK5CYII="
    )


@dataclass
class BenchmarkResult:
    """保存单轮压测的聚合统计结果，便于统一打印与后续汇总。"""

    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_s: float
    qps: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    average_latency_ms: float


class RpcClient:
    """基于原始 TCP socket 的最小 RPC 客户端实现。"""

    def __init__(self, host: str, port: int, inference_pb2) -> None:
        self.host = host
        self.port = port
        self.inference_pb2 = inference_pb2
        self.tag = b"AIR0"
        self._correlation_id = 1
        self._lock = threading.Lock()

    # 发送一次 Predict 请求并返回端到端耗时，单位为毫秒。
    def predict(self, image_bytes: bytes, timeout_ms: int) -> float:
        with self._lock:
            correlation_id = self._correlation_id
            self._correlation_id += 1

        # 组装与 C++ 网关约定一致的请求信封。
        message = self.inference_pb2.GatewayRpcMessage()
        message.message_type = self.inference_pb2.GATEWAY_MESSAGE_TYPE_REQUEST
        message.correlation_id = correlation_id
        message.method_name = "Predict"
        message.predict_request.request_id = str(correlation_id)
        message.predict_request.image_format = "jpeg"
        message.predict_request.timeout_ms = timeout_ms
        message.predict_request.image_bytes = image_bytes

        payload = message.SerializeToString()
        framed = self._encode(payload)

        start = time.perf_counter()
        with socket.create_connection((self.host, self.port), timeout=timeout_ms / 1000) as sock:
            sock.settimeout(timeout_ms / 1000)
            sock.sendall(framed)
            response = self._recv_message(sock)
        end = time.perf_counter()

        # 校验响应类型与业务状态，确保只统计真正成功的请求延迟。
        decoded = self.inference_pb2.GatewayRpcMessage()
        decoded.ParseFromString(response)
        if decoded.message_type != self.inference_pb2.GATEWAY_MESSAGE_TYPE_RESPONSE:
            raise RuntimeError("unexpected response message type")
        if not decoded.HasField("predict_response"):
            raise RuntimeError("predict_response payload is missing")
        if decoded.predict_response.status_code != self.inference_pb2.PREDICT_STATUS_OK:
            raise RuntimeError(decoded.predict_response.error_message)

        return (end - start) * 1000.0

    # 按照网关协议封装长度、标签和校验和，生成可直接发送的帧数据。
    def _encode(self, payload: bytes) -> bytes:
        import zlib

        body = self.tag + payload
        checksum = zlib.adler32(body) & 0xFFFFFFFF
        total_length = len(body) + 4
        return struct.pack(">I", total_length) + body + struct.pack(">I", checksum)

    # 循环读取固定字节数，直到读满或对端关闭连接。
    def _recv_exactly(self, sock: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise RuntimeError("connection closed unexpectedly")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # 从 socket 中读取并校验一帧完整响应，再返回其中的 protobuf 负载。
    def _recv_message(self, sock: socket.socket) -> bytes:
        import zlib

        header = self._recv_exactly(sock, 4)
        total_length = struct.unpack(">I", header)[0]
        body = self._recv_exactly(sock, total_length)
        tag = body[:4]
        if tag != self.tag:
            raise RuntimeError(f"unexpected tag: {tag!r}")
        payload = body[4:-4]
        checksum = struct.unpack(">I", body[-4:])[0]
        actual_checksum = zlib.adler32(tag + payload) & 0xFFFFFFFF
        if checksum != actual_checksum:
            raise RuntimeError("checksum mismatch")
        return payload


# 计算给定分位点，供 P50/P95/P99 延迟统计复用。
def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio) - 1))
    return ordered[index]


# 解析形如 50,200,500,1000 的并发阶梯参数，便于一次完成多轮压测。
def parse_concurrency_levels(value: str) -> List[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels:
        raise ValueError("concurrency levels cannot be empty")
    if any(level <= 0 for level in levels):
        raise ValueError("concurrency levels must be positive integers")
    return levels


# 执行单轮压测并返回聚合指标，供阶梯式压测复用。
def run_benchmark(host: str, port: int, total_requests: int, concurrency: int, image_size: int, timeout_ms: int, inference_pb2) -> BenchmarkResult:
    image_bytes = generate_dummy_image(image_size)
    client = RpcClient(host, port, inference_pb2)
    latencies: List[float] = []
    start = time.perf_counter()
    successes = 0

    # 用线程池并发发起请求，模拟多连接同时访问网关的压力场景。
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(client.predict, image_bytes, timeout_ms) for _ in range(total_requests)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
                successes += 1
            except Exception as exc:
                print(f"Request failed: {exc}")

    duration = time.perf_counter() - start
    return BenchmarkResult(
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=successes,
        failed_requests=total_requests - successes,
        duration_s=duration,
        qps=successes / duration if duration > 0 else 0.0,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        p99_latency_ms=percentile(latencies, 0.99),
        average_latency_ms=statistics.mean(latencies) if latencies else 0.0,
    )


# 以统一格式打印单轮压测结果，便于人工查看和后续解析。
def print_result(result: BenchmarkResult) -> None:
    print(f"Concurrency: {result.concurrency}")
    print(f"Total requests: {result.total_requests}")
    print(f"Successful requests: {result.successful_requests}")
    print(f"Failed requests: {result.failed_requests}")
    print(f"Duration: {result.duration_s:.3f} s")
    print(f"QPS: {result.qps:.2f}")
    print(f"P50 latency: {result.p50_latency_ms:.2f} ms")
    print(f"P95 latency: {result.p95_latency_ms:.2f} ms")
    print(f"P99 latency: {result.p99_latency_ms:.2f} ms")
    print(f"Average latency: {result.average_latency_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test the AI inference gateway with dummy in-memory image bytes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--requests", type=int, default=100, help="Base request count for each round. Actual count is max(requests, concurrency).")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--concurrency-levels", default="", help="Comma-separated stepped concurrency levels, such as 50,200,500,1000.")
    parser.add_argument("--image-size", type=int, default=4096)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    add_project_venv_site_packages(project_root)
    output_dir = build_python_proto(project_root)
    sys.path.insert(0, str(output_dir))
    import inference_pb2  # type: ignore

    # 若指定了并发阶梯，则按档位依次执行多轮压测，并输出整体摘要。
    if args.concurrency_levels:
        results: List[BenchmarkResult] = []
        for concurrency in parse_concurrency_levels(args.concurrency_levels):
            total_requests = max(args.requests, concurrency)
            print(f"=== Benchmark start: concurrency={concurrency}, requests={total_requests} ===")
            result = run_benchmark(args.host, args.port, total_requests, concurrency, args.image_size, args.timeout_ms, inference_pb2)
            print_result(result)
            print("=== Benchmark end ===")
            results.append(result)

        best_qps = max(results, key=lambda item: item.qps)
        best_p99 = min(results, key=lambda item: item.p99_latency_ms)
        print("=== Benchmark summary ===")
        print(f"Peak QPS concurrency: {best_qps.concurrency}, QPS: {best_qps.qps:.2f}")
        print(f"Best P99 concurrency: {best_p99.concurrency}, P99 latency: {best_p99.p99_latency_ms:.2f} ms")
        return

    # 单轮模式下使用 max(requests, concurrency)，避免请求数小于线程数时压测失真。
    total_requests = max(args.requests, args.concurrency)
    result = run_benchmark(args.host, args.port, total_requests, args.concurrency, args.image_size, args.timeout_ms, inference_pb2)
    print_result(result)


if __name__ == "__main__":
    main()
