from __future__ import annotations
import json, socket

class MT5BridgeClient:
    """Client for the EA_SignalBridge.mq5 on 127.0.0.1:18080."""
    def __init__(self, host: str = "127.0.0.1", port: int = 18080):
        self.host, self.port = host, port

    def _tx(self, payload: dict) -> dict:
        data = (json.dumps(payload) + "\n").encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=5) as s:
            s.sendall(data)
            s.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                buf = s.recv(4096)
                if not buf: break
                chunks.append(buf)
        raw = b"".join(chunks).decode("utf-8").strip()
        return json.loads(raw) if raw else {}

    def ping(self) -> dict:
        return self._tx({"cmd": "ping"})

    def place_order(self, symbol: str, side: str, volume: float, sl: float | None = None, tp: float | None = None) -> dict:
        return self._tx({"cmd":"place_order","symbol":symbol,"side":side,"volume":volume,"sl":sl,"tp":tp})

    def close_symbol(self, symbol: str) -> dict:
        return self._tx({"cmd":"close_symbol","symbol":symbol})

    def close_all(self) -> dict:
        return self._tx({"cmd":"close_all"})
