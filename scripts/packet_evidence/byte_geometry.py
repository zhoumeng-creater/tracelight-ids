"""Deterministic first-observed TCP byte geometry; no model or label input."""
from __future__ import annotations
from bisect import bisect_left


def relative(value, base):
    """Signed serial displacement; caller reports spans approaching 2**31."""
    return ((value - base + 2**31) % 2**32) - 2**31


def insert(intervals, lo, hi):
    """Insert into a sorted disjoint union and return previously uncovered pieces."""
    if hi <= lo:
        return []
    i = bisect_left(intervals, (lo,))
    if i and intervals[i-1][1] >= lo:
        i -= 1
    j, cursor, novel = i, lo, []
    newlo, newhi = lo, hi
    while j < len(intervals) and intervals[j][0] <= hi:
        a, b = intervals[j]
        if a > cursor:
            novel.append((cursor, min(a, hi)))
        cursor = max(cursor, b)
        newlo, newhi = min(newlo, a), max(newhi, b)
        j += 1
    if cursor < hi:
        novel.append((cursor, hi))
    intervals[i:j] = [(newlo, newhi)]
    return [(a, b) for a, b in novel if b > a]


class Prefix:
    def __init__(self, intervals):
        self.starts = [a for a, _ in intervals]
        self.ends = [b for _, b in intervals]
        self.cumulative = [0]
        for a, b in intervals:
            self.cumulative.append(self.cumulative[-1] + b - a)
        self.total = self.cumulative[-1]

    def at(self, position):
        i = bisect_left(self.starts, position)
        if i == 0:
            return 0
        return self.cumulative[i-1] + min(position, self.ends[i-1]) - self.starts[i-1]


def analyze(packets):
    """Packets: record, time_ns, direction, seq, ack, flags, declared, captured, IP len.

    This is a full-connection descriptive view, not an early online feature.
    First-observed repeated bytes retain their original ACK annotation.
    """
    bases, syn = [None, None], [False, False]
    for p in packets:
        d, seq, flags = p[2], p[3], p[5]
        if flags & 2 and not syn[d]:
            bases[d], syn[d] = (seq + 1) % 2**32, True
    for p in packets:
        d = p[2]
        if bases[d] is None:
            bases[d] = (p[3] + bool(p[5] & 2)) % 2**32
    unions, pieces, bursts = [[], []], [[], []], []
    packet_counts, declared, captured, first20 = [0, 0], [0, 0], [0, 0], [0, 0]
    duplicate_bytes = payload_packets = pure_ack = reversed_time = ackless = 0
    truncated = large_ip = retransmit_packets = out_of_order = 0
    last_time, highest = None, [None, None]
    for i, p in enumerate(packets):
        rec, time, d, seq, ack, flags, length, cap, ip_length = p
        packet_counts[d] += 1
        declared[d] += length
        captured[d] += cap
        truncated += cap < length
        large_ip += ip_length > 1500
        reversed_time += last_time is not None and time < last_time
        last_time = time
        pure_ack += length == 0 and flags == 16
        if not length:
            continue
        payload_packets += 1
        start = relative((seq + bool(flags & 2)) % 2**32, bases[d])
        end = start + length
        out_of_order += highest[d] is not None and start < highest[d]
        highest[d] = max(highest[d] if highest[d] is not None else end, end)
        novel = insert(unions[d], start, end)
        mass = sum(b-a for a,b in novel)
        duplicate_bytes += length - mass
        retransmit_packets += mass < length
        if i < 20:
            first20[d] += mass
        ackpos = relative(ack, bases[1-d]) if flags & 16 and bases[1-d] is not None else None
        ackless += bool(mass) and ackpos is None
        for a, b in novel:
            pieces[d].append((a, b, ackpos, rec, time))
        if mass:
            if bursts and bursts[-1][0] == d:
                bursts[-1][1] += mass
            else:
                bursts.append([d, mass])
    prefixes = [Prefix(u) for u in unions]
    totals = [v.total for v in prefixes]
    segments, ack_queries = [], [set(), set()]
    for d in (0, 1):
        out = []
        for a, b, ackpos, rec, time in sorted(pieces[d]):
            covered = prefixes[1-d].at(ackpos) if ackpos is not None else None
            # Coordinates count observed unique bytes; gaps remain a separate quality fact.
            left, right = prefixes[d].at(a), prefixes[d].at(b)
            if covered is not None:
                ack_queries[1-d].add(covered)
            if out and out[-1][1] == left and out[-1][2] == covered:
                out[-1][1] = right
            else:
                out.append([left, right, covered])
        segments.append(out)
    denominator = totals[0] * totals[1]
    coverage = (sum((b-a)*ack for ss in segments for a,b,ack in ss if ack is not None)
                / denominator if denominator else None)
    gap_bytes = [u[-1][1] - u[0][0] - total if u else 0 for u, total in zip(unions, totals)]
    missing_initial = [bool(u) and syn[d] and u[0][0] != 0 for d,u in enumerate(unions)]
    eligible = all(syn) and all(totals) and not any(gap_bytes) and not any(missing_initial) and not ackless
    # Strong simple control: serial direction-burst byte totals imply an ACK
    # frontier if every new burst has acknowledged all earlier opposite bytes.
    burst_segments, progress = [[], []], [0, 0]
    for d, mass in bursts:
        burst_segments[d].append([progress[d], progress[d]+mass, progress[1-d]])
        progress[d] += mass
    for d in (0, 1):
        merged = []
        for a,b,ack in burst_segments[d]:
            if merged and merged[-1][1]==a and merged[-1][2]==ack:
                merged[-1][1]=b
            else:
                merged.append([a,b,ack])
        burst_segments[d]=merged
    return {
        'packet_count': len(packets), 'packet_counts': packet_counts,
        'payload_packets': payload_packets, 'pure_ack_packets': pure_ack,
        'declared_payload_bytes': declared, 'captured_payload_bytes': captured,
        'unique_payload_bytes': totals, 'duplicate_payload_bytes': duplicate_bytes,
        'duplicate_fraction': duplicate_bytes / sum(declared) if sum(declared) else None,
        'repeated_byte_packets': retransmit_packets, 'lower_sequence_packets': out_of_order,
        'truncated_payload_packets': truncated, 'large_ip_packets': large_ip,
        'timestamp_reversals': reversed_time, 'syn_seen': syn,
        'gap_bytes': gap_bytes, 'missing_initial_bytes': missing_initial,
        'data_with_unanchored_ack': ackless,
        'geometry_eligible': bool(eligible),
        'first_observed_piece_count': sum(map(len, pieces)),
        'ack_segment_count': sum(map(len, segments)),
        'ack_query_count': sum(map(len, ack_queries)),
        'direction_burst_count': len(bursts), 'bursts': bursts,
        'ack_covered_pair_fraction': coverage,
        'ack_map_exactly_reconstructed_from_direction_bursts': segments == burst_segments,
        'first20_unique_byte_fraction': sum(first20)/sum(totals) if sum(totals) else None,
        'segments': segments,
        'serial_span_ambiguous': any(u and u[-1][1]-u[0][0] >= 2**31 for u in unions),
    }
