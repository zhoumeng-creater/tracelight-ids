"""R21's four causal counts in one sequential pass, including ties across batches."""

from collections import Counter, deque

import numpy as np

from ids.flow_history.features import WINDOW_MS


def dictionary(column):
    values = column.combine_chunks().dictionary_encode()
    return values.dictionary.to_pylist(), values.indices.to_numpy()


class EventBlock:
    """Compact arrays and block-local address dictionaries for the active window."""

    def __init__(self, table):
        self.time = table["flow_end_ms"].to_numpy()
        sources, src = dictionary(table["source"])
        protocols, proto = dictionary(table["native_protocol"])
        keys, self.key = np.unique(src.astype(np.int64) * len(protocols) + proto, return_inverse=True)
        self.keys = [(sources[k // len(protocols)], protocols[k % len(protocols)]) for k in keys]
        self.key = self.key.astype(np.int32)
        self.destinations, self.dst = dictionary(table["destination"])
        self.port = table["resp_port"].to_numpy().astype(np.int32)
        replies = table["resp_pkts"].to_numpy()
        self.reply = np.sign(replies).astype(np.int8)
        self.first = 0


class ActiveGroup:
    __slots__ = ("rows", "known", "unanswered", "destinations", "ports")

    def __init__(self):
        self.rows = self.known = self.unanswered = 0
        self.destinations = Counter()
        self.ports = Counter()

    def snapshot(self):
        fraction = self.unanswered / self.known if self.known else (-1.0 if self.rows else 0.0)
        return self.rows, len(self.destinations), len(self.ports), fraction

    def add(self, destination, port, reply):
        self.rows += 1
        self.known += int(reply >= 0)
        self.unanswered += int(reply == 0)
        self.destinations[destination] += 1
        if port >= 0:
            self.ports[port] += 1

    def discard(self, destination, port, reply):
        self.rows -= 1
        self.known -= int(reply >= 0)
        self.unanswered -= int(reply == 0)
        self.destinations[destination] -= 1
        if not self.destinations[destination]:
            del self.destinations[destination]
        if port >= 0:
            self.ports[port] -= 1
            if not self.ports[port]:
                del self.ports[port]


class SourceHistory:
    def __init__(self):
        self.blocks = deque()
        self.groups = {}
        self.tied = {}
        self.watermark = -1

    def expire(self, cutoff):
        while self.blocks:
            block = self.blocks[0]
            i = block.first
            if block.time[i] >= cutoff:
                return
            while i < len(block.time) and block.time[i] < cutoff:
                key = block.keys[block.key[i]]
                group = self.groups[key]
                group.discard(block.destinations[block.dst[i]], int(block.port[i]), int(block.reply[i]))
                if not group.rows:
                    del self.groups[key]
                i += 1
            if i == len(block.time):
                self.blocks.popleft()
            else:
                block.first = i
                return

    def transform(self, table):
        block = EventBlock(table)
        if block.time[0] < self.watermark or (np.diff(block.time) < 0).any():
            raise ValueError("Zeek logical time decreased between observations")
        self.blocks.append(block)
        counts = np.empty((len(table), 3), np.int32)
        fraction = np.empty(len(table), np.float64)
        for i, time in enumerate(block.time):
            if time != self.watermark:
                self.expire(time - WINDOW_MS)
                self.watermark = time
                self.tied.clear()
            key = block.keys[block.key[i]]
            group = self.groups.get(key)
            if group is None:
                self.groups[key] = group = ActiveGroup()
            values = self.tied.get(key)
            if values is None:
                # The first snapshot at t stays valid for every other record at t.
                self.tied[key] = values = group.snapshot()
            counts[i] = values[:3]
            fraction[i] = values[3]
            group.add(block.destinations[block.dst[i]], int(block.port[i]), int(block.reply[i]))
        return (*counts.T, fraction)
