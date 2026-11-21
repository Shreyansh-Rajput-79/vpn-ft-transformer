"""PCAP -> ISCX time-based flow features.

Re-implements the ISCX VPN-nonVPN 2016 feature definitions so that flows from a
capture look like the rows the model was trained on. NFStream is not used here
because it reports packet times in milliseconds, while the training features are
in microseconds (median min_fiat is ~60 us).

Parameters below were read off the training data:
  * flows are cut 15 s after their first packet (30% of flows are exactly 15 s),
  * a gap over 1 s starts an idle period (no idle period in the data is < 1 s),
  * byte rates use IP-layer packet length (smallest bytes/packet is 30-40),
  * absent statistics are min/max = -1, mean/std = 0.
"""
import socket

import dpkt
import numpy as np
import pandas as pd

from utils.preprocessing import FEATURES, NFSTREAM_FEATURES

FLOW_TIMEOUT_US = 15_000_000
ACTIVITY_TIMEOUT_US = 1_000_000

_DLT_NULL, _DLT_EN10MB, _DLT_RAW, _DLT_LOOP, _DLT_LINUX_SLL = 0, 1, 101, 108, 113
_RAW_TYPES = {12, 14, _DLT_RAW, 228, 229}


class _Flow:
    __slots__ = ("src", "dst", "proto", "times", "forward", "sizes")

    def __init__(self, src, dst, proto):
        self.src, self.dst, self.proto = src, dst, proto
        self.times, self.forward, self.sizes = [], [], []

    def add(self, t, forward, size):
        self.times.append(t)
        self.forward.append(forward)
        self.sizes.append(size)


def _open(f):
    try:
        return dpkt.pcap.Reader(f)
    except ValueError:
        f.seek(0)
        return dpkt.pcapng.Reader(f)


def _ip_layer(buf, linktype):
    try:
        if linktype == _DLT_EN10MB:
            ip = dpkt.ethernet.Ethernet(buf).data
        elif linktype == _DLT_LINUX_SLL:
            ip = dpkt.sll.SLL(buf).data
        elif linktype in (_DLT_NULL, _DLT_LOOP):
            ip = dpkt.loopback.Loopback(buf).data
        elif linktype in _RAW_TYPES:
            ip = dpkt.ip6.IP6(buf) if buf[0] >> 4 == 6 else dpkt.ip.IP(buf)
        else:
            raise ValueError(f"Unsupported link type: {linktype}")
    except (dpkt.UnpackError, IndexError):
        return None
    return ip if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) else None


def _addr(ip, raw):
    family = socket.AF_INET6 if isinstance(ip, dpkt.ip6.IP6) else socket.AF_INET
    return socket.inet_ntop(family, raw)


def _stats(values):
    """(min, mean, max, std) with the ISCX convention for an empty list."""
    if not values:
        return -1, 0.0, -1, 0.0
    a = np.asarray(values, dtype=np.float64)
    std = a.std(ddof=1) if len(a) > 1 else 0.0
    return a.min(), a.mean(), a.max(), std


def _active_idle(times):
    active, idle = [], []
    start = end = times[0]
    for t in times[1:]:
        if t - end > ACTIVITY_TIMEOUT_US:
            if end - start > 0:
                active.append(end - start)
            idle.append(t - end)
            start = end = t
        else:
            end = t
    # In the training data active periods only appear alongside idle ones.
    if idle and end - start > 0:
        active.append(end - start)
    return active, idle


def _flow_features(flow):
    times = flow.times
    fwd = [t for t, f in zip(times, flow.forward) if f]
    bwd = [t for t, f in zip(times, flow.forward) if not f]

    duration = times[-1] - times[0]
    seconds = duration / 1e6

    min_f, mean_f, max_f, std_f = _stats(np.diff(fwd).tolist())
    min_b, mean_b, max_b, std_b = _stats(np.diff(bwd).tolist())
    min_iat, mean_iat, max_iat, std_iat = _stats(np.diff(times).tolist())
    active, idle = _active_idle(times)
    min_a, mean_a, max_a, std_a = _stats(active)
    min_i, mean_i, max_i, std_i = _stats(idle)

    row = {
        "src_ip": flow.src[0], "src_port": flow.src[1],
        "dst_ip": flow.dst[0], "dst_port": flow.dst[1],
        "protocol": "TCP" if flow.proto == dpkt.ip.IP_PROTO_TCP else "UDP",
        "start_time": pd.to_datetime(times[0], unit="us"),
        "packets": len(times), "bytes": sum(flow.sizes),
        "duration": duration,
        "min_fiat": min_f, "max_fiat": max_f, "mean_fiat": mean_f, "std_fiat": std_f,
        "min_biat": min_b, "max_biat": max_b, "mean_biat": mean_b, "std_biat": std_b,
        "flowPktsPerSecond": len(times) / seconds if seconds > 0 else 0.0,
        "flowBytesPerSecond": sum(flow.sizes) / seconds if seconds > 0 else 0.0,
        "min_flowiat": min_iat, "max_flowiat": max_iat,
        "mean_flowiat": mean_iat, "std_flowiat": std_iat,
        "min_active": min_a, "mean_active": mean_a, "max_active": max_a, "std_active": std_a,
        "min_idle": min_i, "mean_idle": mean_i, "max_idle": max_i, "std_idle": std_i,
    }
    return row


def pcap_to_df(pcap_file):
    """Split a .pcap/.pcapng into bidirectional TCP/UDP flows and compute ISCX features.

    Returns one row per flow: flow identifiers plus every column in FEATURES.
    """
    active = {}
    rows = []

    with open(pcap_file, "rb") as f:
        reader = _open(f)
        linktype = reader.datalink()

        for i, (ts, buf) in enumerate(reader):
            ip = _ip_layer(buf, linktype)
            if ip is None or not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                continue

            t = int(round(ts * 1e6))
            l4 = ip.data
            src = (_addr(ip, ip.src), l4.sport)
            dst = (_addr(ip, ip.dst), l4.dport)
            proto = ip.p if isinstance(ip, dpkt.ip.IP) else ip.nxt
            size = ip.len if isinstance(ip, dpkt.ip.IP) else ip.plen + 40

            key = (proto,) + ((src, dst) if src <= dst else (dst, src))
            flow = active.get(key)
            if flow is not None and t - flow.times[0] > FLOW_TIMEOUT_US:
                rows.append(_flow_features(active.pop(key)))
                flow = None
            if flow is None:
                # The first packet's sender defines the forward direction.
                flow = active[key] = _Flow(src, dst, proto)
            flow.add(t, src == flow.src, size)

            # Periodically flush flows that can no longer receive packets.
            if i % 10_000 == 0:
                for k in [k for k, fl in active.items() if t - fl.times[0] > FLOW_TIMEOUT_US]:
                    rows.append(_flow_features(active.pop(k)))

    rows.extend(_flow_features(fl) for fl in active.values())

    columns = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol",
               "start_time", "packets", "bytes"] + FEATURES
    return pd.DataFrame(rows, columns=columns).sort_values("start_time", ignore_index=True)


def pcap_to_df_nfstream(pcap_file, idle_timeout=120, active_timeout=1800):
    """Split a capture into flows with NFStream and keep its behavioural features.

    Returns the same flow identifiers as pcap_to_df plus every column in
    NFSTREAM_FEATURES. Identifier columns are for reporting only; the model
    never sees them.
    """
    from nfstream import NFStreamer

    df = NFStreamer(source=str(pcap_file), statistical_analysis=True,
                    idle_timeout=idle_timeout, active_timeout=active_timeout).to_pandas()
    if df.empty:
        return pd.DataFrame(columns=["src_ip", "src_port", "dst_ip", "dst_port", "protocol",
                                     "start_time", "packets", "bytes"] + NFSTREAM_FEATURES)

    meta = pd.DataFrame({
        "src_ip": df.src_ip, "src_port": df.src_port,
        "dst_ip": df.dst_ip, "dst_port": df.dst_port,
        "protocol": df.protocol.map({6: "TCP", 17: "UDP"}).fillna(df.protocol.astype(str)),
        "start_time": pd.to_datetime(df.bidirectional_first_seen_ms, unit="ms"),
        "packets": df.bidirectional_packets, "bytes": df.bidirectional_bytes,
    })
    return pd.concat([meta, df[NFSTREAM_FEATURES]], axis=1)


EXTRACTORS = {"iscx": pcap_to_df, "nfstream": pcap_to_df_nfstream}
