import argparse
import base64
import os
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
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


def generate_dummy_image(size: int) -> bytes:
    del size
    # 使用内存中构造的 1x1 PNG 作为稳定的 Dummy Data，避免随机字节无法被服务端解码。
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0JYAAAAASUVORK5CYII="
    )


class RpcClient:
    def __init__(self, host: str, port: int, inference_pb2) -> None:
        self.host = host
        self.port = port
        self.inference_pb2 = inference_pb2
        self.tag = b"AIR0"
        self._correlation_id = 1
        self._lock = threading.Lock()

    def predict(self, image_bytes: bytes, timeout_ms: int) -> float:
        with self._lock:
            correlation_id = self._correlation_id
            self._correlation_id += 1

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

        decoded = self.inference_pb2.GatewayRpcMessage()
        decoded.ParseFromString(response)
        if decoded.message_type != self.inference_pb2.GATEWAY_MESSAGE_TYPE_RESPONSE:
            raise RuntimeError("unexpected response message type")
        if not decoded.HasField("predict_response"):
            raise RuntimeError("predict_response payload is missing")
        if decoded.predict_response.status_code != self.inference_pb2.PREDICT_STATUS_OK:
            raise RuntimeError(decoded.predict_response.error_message)

        return (end - start) * 1000.0

    def _encode(self, payload: bytes) -> bytes:
        import zlib

        body = self.tag + payload
        checksum = zlib.adler32(body) & 0xFFFFFFFF
        total_length = len(body) + 4
        return struct.pack(">I", total_length) + body + struct.pack(">I", checksum)

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


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test the AI inference gateway with dummy in-memory image bytes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=4096)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    add_project_venv_site_packages(project_root)
    output_dir = build_python_proto(project_root)
    sys.path.insert(0, str(output_dir))
    import inference_pb2  # type: ignore

    image_bytes = generate_dummy_image(args.image_size)
    client = RpcClient(args.host, args.port, inference_pb2)

    latencies: List[float] = []
    start = time.perf_counter()
    successes = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(client.predict, image_bytes, args.timeout_ms) for _ in range(args.requests)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
                successes += 1
            except Exception as exc:
                print(f"Request failed: {exc}")

    duration = time.perf_counter() - start
    qps = successes / duration if duration > 0 else 0.0

    print(f"Total requests: {args.requests}")
    print(f"Successful requests: {successes}")
    print(f"Failed requests: {args.requests - successes}")
    print(f"Duration: {duration:.3f} s")
    print(f"QPS: {qps:.2f}")
    print(f"P50 latency: {percentile(latencies, 0.50):.2f} ms")
    print(f"P95 latency: {percentile(latencies, 0.95):.2f} ms")
    print(f"P99 latency: {percentile(latencies, 0.99):.2f} ms")
    if latencies:
        print(f"Average latency: {statistics.mean(latencies):.2f} ms")


if __name__ == "__main__":
    main()
