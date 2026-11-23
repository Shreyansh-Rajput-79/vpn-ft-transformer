"""A small VPN-style tunnel for demos: real traffic, encrypted, carried over UDP.

    app (e.g. curl) --TCP--> tunnel client --encrypted UDP--> tunnel server --TCP--> real server

Both ends run on this machine, so the tunnel crosses the loopback interface, where it
can be captured live (Windows: Npcap with loopback support, device \\Device\\NPF_Loopback).

Every tunnel packet looks like an OpenVPN data packet: a header (opcode, session id,
packet id), a random nonce, the encrypted payload and an authentication tag, and the
client sends a keep-alive ping every 10 s (OpenVPN's default). Encryption is a SHA-256
keystream with an HMAC-SHA256 tag (standard library only).

It is NOT a full VPN: it tunnels application data per TCP connection (like a proxy),
not whole IP packets, so the inner TCP acknowledgements are not carried. Use it to show
that encrypted, encapsulated traffic is recognised, not as a benchmark.

Usage (three terminals, or see the README):
    python tools/demo_tunnel.py --remote proof.ovh.net:443 --listen 127.0.0.1:443
    python predict_live.py -i \\Device\\NPF_Loopback --bpf "udp port 1195" -m models/vnat_noflags_model.pth --trace 1
    curl --resolve proof.ovh.net:443:127.0.0.1 https://proof.ovh.net/files/10Mb.dat -o NUL --max-time 25
"""
import argparse
import hashlib
import hmac
import os
import socket
import struct
import threading
import time

KEY = hashlib.sha256(b"demo tunnel key").digest()
OPEN, DATA, CLOSE, PING = 1, 2, 3, 4
HEADER = struct.Struct("!B8sIBH")       # opcode, session id, packet id, type, connection id
NONCE, TAG, CHUNK = 12, 16, 1300


def _keystream(nonce, n):
    out = bytearray()
    for counter in range((n + 31) // 32):
        out += hashlib.sha256(KEY + nonce + counter.to_bytes(4, "big")).digest()
    return bytes(out[:n])


class Codec:
    def __init__(self):
        self.session, self.packet_id, self.lock = os.urandom(8), 0, threading.Lock()

    def seal(self, kind, conn, payload=b""):
        with self.lock:
            self.packet_id += 1
            header = HEADER.pack(0x48, self.session, self.packet_id, kind, conn)
        nonce = os.urandom(NONCE)
        body = bytes(a ^ b for a, b in zip(payload, _keystream(nonce, len(payload))))
        tag = hmac.new(KEY, header + nonce + body, hashlib.sha256).digest()[:TAG]
        return header + nonce + body + tag

    @staticmethod
    def open(packet):
        header, nonce = packet[:HEADER.size], packet[HEADER.size:HEADER.size + NONCE]
        body, tag = packet[HEADER.size + NONCE:-TAG], packet[-TAG:]
        if not hmac.compare_digest(tag, hmac.new(KEY, header + nonce + body, hashlib.sha256).digest()[:TAG]):
            return None
        _, _, _, kind, conn = HEADER.unpack(header)
        return kind, conn, bytes(a ^ b for a, b in zip(body, _keystream(nonce, len(body))))


RATE = 300_000   # bytes/s per connection: UDP has no flow control, so pace the sender


def pump(sock, send, conn, codec, rate=None):
    """TCP socket -> encrypted UDP datagrams (paced to `rate` bytes/s if given)."""
    try:
        while data := sock.recv(CHUNK):
            send(codec.seal(DATA, conn, data))
            if rate:
                time.sleep(len(data) / rate)
    except OSError:
        pass
    send(codec.seal(CLOSE, conn))


def server(port, remote):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    udp.bind(("127.0.0.1", port))
    codec, conns = Codec(), {}
    host, rport = remote.rsplit(":", 1)
    while True:
        packet, peer = udp.recvfrom(65535)
        msg = Codec.open(packet)
        if msg is None:
            continue
        kind, conn, payload = msg
        send = lambda p, peer=peer: udp.sendto(p, peer)
        if kind == OPEN:
            s = socket.create_connection((host, int(rport)))
            conns[conn] = s
            threading.Thread(target=pump, args=(s, send, conn, codec, RATE), daemon=True).start()
        elif kind == DATA and conn in conns:
            conns[conn].sendall(payload)
        elif kind == CLOSE and conn in conns:
            conns.pop(conn).close()
        elif kind == PING:
            send(codec.seal(PING, 0, b"\x2a" * 16))


def client(listen, port):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    udp.connect(("127.0.0.1", port))
    codec, conns, next_id = Codec(), {}, 1
    send = udp.send

    def downstream():
        while True:
            msg = Codec.open(udp.recv(65535))
            if msg is None:
                continue
            kind, conn, payload = msg
            if kind == DATA and conn in conns:
                conns[conn].sendall(payload)
            elif kind == CLOSE and conn in conns:
                conns.pop(conn).close()

    def keepalive():
        while True:
            time.sleep(10)
            send(codec.seal(PING, 0, b"\x2a" * 16))

    threading.Thread(target=downstream, daemon=True).start()
    threading.Thread(target=keepalive, daemon=True).start()
    host, lport = listen.rsplit(":", 1)
    tcp = socket.socket()
    tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp.bind((host, int(lport)))
    tcp.listen()
    print(f"[tunnel] apps connect to {listen}; tunnel = UDP 127.0.0.1:{port}", flush=True)
    while True:
        s, _ = tcp.accept()
        conn, next_id = next_id, next_id + 1
        conns[conn] = s
        send(codec.seal(OPEN, conn))
        time.sleep(0.05)
        threading.Thread(target=pump, args=(s, send, conn, codec), daemon=True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", required=True, help="real server the tunnel exit connects to, host:port")
    ap.add_argument("--listen", default="127.0.0.1:443", help="where applications connect, host:port")
    ap.add_argument("--port", type=int, default=1195, help="UDP port of the tunnel")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    args = ap.parse_args()
    threading.Thread(target=server, args=(args.port, args.remote), daemon=True).start()
    time.sleep(0.2)
    threading.Thread(target=client, args=(args.listen, args.port), daemon=True).start()
    time.sleep(args.seconds or 1e9)
