"""Count association/clock differences on extracted headers, without rereading PCAP."""
from collections import Counter
import argparse
import gzip
import json
from pathlib import Path


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def check(receipt):
    base = receipt.parent
    name = receipt.name.removesuffix('.receipt.json')
    cs = list(rows(base / (name + '.connections.jsonl.gz')))
    all_counts = [[0, 0] for c in cs]
    single_counts = [[0, 0] for c in cs]
    observed_counts = [[0, 0] for c in cs]
    counted_ip = [[0, 0] for c in cs]
    extras = Counter()
    clock = Counter()
    candidate_flows, ambiguous_flows = set(), set()
    max_after_export, max_after_observed = 0, 0
    for p in rows(base / (name + '.jsonl.gz')):
        indices = p['connection_indices']
        observed = p['observed_interval_connection_indices']
        for i in indices:
            c = cs[i]
            direction = 0 if (p['source'], p['source_port']) == (c['source'], c['raw8']['orig_port']) else 1
            candidate_flows.add(i)
            all_counts[i][direction] += 1
            counted_ip[i][direction] += p['ip_total_bytes']
            if len(indices) == 1:
                single_counts[i][direction] += 1
            else:
                ambiguous_flows.add(i)
            t = p['timestamp_us']
            end = round((c['zeek_reported_end'] or c['connection_start']) * 1e6)
            maximum = max(0, t-end)
            max_after_observed = max(max_after_observed, maximum)
            if c['replay_export_time'] is not None:
                delta = t-round(c['replay_export_time'] * 1e6)
                clock['candidate_associations_after_export_any'] += delta > 0
                clock['candidate_associations_after_export_over_10us'] += delta > 10
                max_after_export = max(max_after_export, delta)
            else:
                clock['candidate_associations_without_export_clock'] += 1
            clock['candidate_associations_before_start_any'] += t < round(c['connection_start'] * 1e6)
        for i in observed:
            c = cs[i]
            direction = 0 if (p['source'], p['source_port']) == (c['source'], c['raw8']['orig_port']) else 1
            observed_counts[i][direction] += 1
        if indices and not observed:
            extras['records'] += 1
            extras['zero_declared_payload'] += p['declared_tcp_payload_bytes'] == 0
            extras['positive_declared_payload'] += p['declared_tcp_payload_bytes'] > 0
            extras['ack_flag'] += bool(p['flags'] & 16)
            extras['fin_flag'] += bool(p['flags'] & 1)
            extras['rst_flag'] += bool(p['flags'] & 4)
            extras['syn_flag'] += bool(p['flags'] & 2)
            extras['pure_ack_flags_zero_payload'] += p['flags'] == 16 and p['declared_tcp_payload_bytes'] == 0
    comparison = Counter()
    differences = []
    for i in sorted(candidate_flows):
        c = cs[i]
        raw = c['raw8']
        expected = [raw['orig_pkts'], raw['resp_pkts']]
        expected_ip = [raw['orig_ip_bytes'], raw['resp_ip_bytes']]
        if any(v is None or v < 0 for v in expected):
            comparison['mapped_flows_missing_zeek_packet_counts'] += 1
            continue
        comparison['mapped_flows_with_zeek_packet_counts'] += 1
        comparison['all_candidate_packet_counts_equal_zeek_both_directions'] += all_counts[i] == expected
        comparison['singleton_packet_counts_equal_zeek_both_directions'] += single_counts[i] == expected
        comparison['observed_interval_packet_counts_equal_zeek_both_directions'] += observed_counts[i] == expected
        comparison['all_candidate_ip_byte_counts_equal_zeek_both_directions'] += counted_ip[i] == expected_ip
        comparison['zeek_packets_both_directions_sum'] += sum(expected)
        comparison['all_candidate_packets_both_directions_sum'] += sum(all_counts[i])
        comparison['singleton_packets_both_directions_sum'] += sum(single_counts[i])
        comparison['observed_interval_packets_both_directions_sum'] += sum(observed_counts[i])
        if all_counts[i] != expected:
            comparison['packet_difference_has_ambiguous_membership'] += i in ambiguous_flows
            comparison['packet_difference_all_candidate_ge_zeek_both_directions'] += all(a >= b for a,b in zip(all_counts[i], expected))
            differences.append(dict(connection_index=i, flow_id=c['flow_id'], stable_flow_id=c.get('stable_flow_id'),
                                    all_candidate=all_counts[i], singleton=single_counts[i], observed=observed_counts[i],
                                    zeek=expected, ambiguous_membership=i in ambiguous_flows,
                                    missed_bytes=c['missed_bytes'], conn_state=c['conn_state']))
    return dict(source=name, lifecycle_only=dict(extras), clock=dict(clock),
                max_candidate_us_after_export=max_after_export, max_candidate_us_after_observed_end=max_after_observed,
                mapped_flows=len(candidate_flows), ambiguous_membership_flows=len(ambiguous_flows),
                packet_count_comparison=dict(comparison), all_candidate_packet_count_differences=differences)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--headers', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    result=[]
    for receipt in sorted(args.headers.glob('*.receipt.json')):
        item=check(receipt)
        result.append(item)
        print(json.dumps({k:v for k,v in item.items() if k != 'all_candidate_packet_count_differences'}), flush=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
