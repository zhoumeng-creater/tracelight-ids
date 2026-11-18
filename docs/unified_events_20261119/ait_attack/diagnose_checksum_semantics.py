"""Read-only checksum audit of fixed R27 control/download packet candidates.

No packet editing, new Zeek export, model call, or frozen-output mutation.
Reuses the cloud-side RFC checksum implementation and original R27 PCAP reader.
"""
from collections import Counter, defaultdict
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parents[2]
ARCHIVE = THESIS / '实验归档/20261116_第二十七轮独立事件取证/回传/packet'
REPO = THESIS / '代码/tracelight-ids/.worktrees/ait-flow-contract-20261118'
HOST = '10.143.2.4'
DIRECTIONS = ('host_to_remote', 'remote_to_host')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


checksum = module('cloud_checksum_helpers', ROOT.parent / 'ait/diagnose_checksum_semantics.py')
reader = module('r27_packet_reader', REPO / 'scripts/prepare_ait_packet_evidence.py')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def endpoint_key(a, ap, b, bp):
    return tuple(sorted(((a, int(ap)), (b, int(bp)))))


def main():
    assert (~checksum.folded_sum(bytes.fromhex('0001f203f4f5f6f7')) & 65535) == 0x220D
    controls = [x for x in csv_rows(ARCHIVE / 'action-connection-candidates.csv') if x['kind'] == 'webshell_command_request']
    downloads = [x for x in csv_rows(ARCHIVE / 'victim-download-connection-candidates.csv') if x['connection_id'] in ('3574', '3575', '3578')]
    assert len(controls) == 32 and len(downloads) == 3
    actions = defaultdict(list)
    roles = defaultdict(set)
    for a in controls:
        for cid in a['candidate_connection_ids'].split(';'):
            actions[int(cid)].append(a['action_id'])
            roles[int(cid)].add('webshell_control')
    for a in downloads:
        actions[int(a['connection_id'])].append(a['download_name'])
        roles[int(a['connection_id'])].add('attack_download')
    catalog = {int(x['connection_id']): x for x in csv_rows(ARCHIVE / 'tcp-connection-candidates.csv') if int(x['connection_id']) in actions}
    assert len(catalog) == 36
    frozen_paths = list((ROOT / 'contract').rglob('*'))
    frozen_paths = [p for p in frozen_paths if p.is_file()]
    before = {str(p.relative_to(ROOT)): checksum.sha(p) for p in frozen_paths}
    pcap = ROOT / 'source/log.pcap.1642684613'
    pcap_sha = checksum.sha(pcap)
    assert pcap_sha == '12c798074e27969ea40719700ecc8cb07ecda76e4f0751d746e02e2398067790'
    by_key = defaultdict(list)
    stats, handshakes, packet_samples, count_orig = {}, {}, defaultdict(list), defaultdict(Counter)
    for cid, p in catalog.items():
        by_key[endpoint_key(p['origin_ip'], p['origin_port'], p['responder_ip'], p['responder_port'])].append(cid)
        stats[cid] = {d: Counter() for d in DIRECTIONS}
        handshakes[cid] = {'syn': None, 'syn_ack': None, 'final_ack': None, 'max_origin_acknowledged_responder_sequence_bytes': 0, 'max_responder_acknowledged_origin_sequence_bytes': 0}
    duplicate_assignments = 0
    for number, (timestamp, linktype, frame, wire) in enumerate(reader.pcap_records(pcap), 1):
        parsed = checksum.ipv4(frame, linktype)
        if parsed is None:
            continue
        ip, tcp, meta = parsed
        ids = by_key.get(endpoint_key(meta['source_ip'], meta['source_port'], meta['destination_ip'], meta['destination_port']), [])
        ids = [cid for cid in ids if float(catalog[cid]['first']) <= timestamp <= float(catalog[cid]['last'])]
        if not ids:
            continue
        duplicate_assignments += len(ids) > 1
        check = checksum.checksum_info(ip, tcp, meta)
        direction = 'host_to_remote' if meta['source_ip'] == HOST else 'remote_to_host'
        event = {'packet_number': number, 'timestamp': timestamp, 'direction': direction, 'pcap_captured_frame_bytes':len(frame), 'pcap_original_frame_bytes':wire, **meta, **check}
        for cid in ids:
            p = catalog[cid]
            origin = meta['source_ip'] == p['origin_ip'] and meta['source_port'] == int(p['origin_port'])
            count_orig[cid]['origin' if origin else 'responder'] += 1
            c = stats[cid][direction]
            c['packets'] += 1
            c['ipv4_declared_bytes_all'] += meta['ip_total_length']
            if check['tcp_checksum'] == 'correct' and check['ipv4_checksum'] == 'correct':
                c['ipv4_declared_bytes_checksums_correct'] += meta['ip_total_length']
                c['both_checksums_correct'] += 1
            c['pcap_captured_less_than_original'] += len(frame) < wire
            c['ipv4_checksum_' + check['ipv4_checksum']] += 1
            c['tcp_checksum_' + check['tcp_checksum']] += 1
            c['tcp_checksum_verifiable'] += check['tcp_checksum'] != 'unverifiable'
            if check['tcp_checksum'] != 'unverifiable':
                c['ipv4_declared_bytes_tcp_verifiable'] += meta['ip_total_length']
            if check['tcp_unverifiable_reason']:
                c['unverifiable_' + check['tcp_unverifiable_reason']] += 1
            c['ipv4_total_exceeds_captured'] += check['ipv4_total_exceeds_captured']
            c['ipv4_total_zero'] += check['ipv4_total_zero']
            c['fragmented'] += meta['fragment_bits'] != 0
            c['tcp_checksum_matches_pseudoheader_seed'] += check['tcp_checksum_matches_pseudoheader_seed'] is True
            c['tcp_incorrect_matches_pseudoheader_seed'] += check['tcp_checksum'] == 'incorrect' and check['tcp_checksum_matches_pseudoheader_seed'] is True
            sample_key = (direction, check['tcp_checksum'], check['ipv4_checksum'])
            existing_keys = {(x['direction'],x['tcp_checksum'],x['ipv4_checksum']) for x in packet_samples[cid]}
            if (check['tcp_checksum'] != 'correct' or check['ipv4_checksum'] != 'correct') and sample_key not in existing_keys:
                packet_samples[cid].append(event)
            h = handshakes[cid]
            flags = meta['flags']
            if origin and flags & 2 and not flags & 16 and h['syn'] is None:
                h['syn'] = event
            elif not origin and flags & 2 and flags & 16 and h['syn'] and meta['acknowledgment'] == (h['syn']['sequence'] + 1) % 2**32 and h['syn_ack'] is None:
                h['syn_ack'] = event
            if origin and flags & 16 and not flags & 2 and h['syn_ack']:
                delta = (meta['acknowledgment'] - h['syn_ack']['sequence'] - 1) % 2**32
                if delta < 2**31:
                    h['max_origin_acknowledged_responder_sequence_bytes'] = max(h['max_origin_acknowledged_responder_sequence_bytes'], delta)
                if meta['acknowledgment'] == (h['syn_ack']['sequence'] + 1) % 2**32 and meta['sequence'] == (h['syn']['sequence'] + 1) % 2**32 and h['final_ack'] is None:
                    h['final_ack'] = event
            if not origin and flags & 16 and h['syn']:
                delta = (meta['acknowledgment'] - h['syn']['sequence'] - 1) % 2**32
                if delta < 2**31:
                    h['max_responder_acknowledged_origin_sequence_bytes'] = max(h['max_responder_acknowledged_origin_sequence_bytes'], delta)
    assert duplicate_assignments == 0
    arms = {}
    arm_paths = {}
    for arm in ('strict', 'ignore_checksums'):
        folders = list((ROOT / f'contract/analysis-{arm}').glob('attempt-*'))
        assert len(folders) == 1
        arm_paths[arm] = folders[0]
        arms[arm] = csv_rows(folders[0] / 'raw8.csv')
    facts_path = ROOT / 'source/capture_config/gather/intranet_server/facts.json'
    facts = json.loads(facts_path.read_text(encoding='utf-8'))
    facts_selected = {k: facts[k] for k in ('ansible_date_time', 'ansible_default_ipv4', 'ansible_kernel', 'suricata_interface', 'suricata_listenmode')}
    facts_selected['interface_features'] = facts['ansible_ens3']['features']
    facts_selected['interface_module'] = facts['ansible_ens3']['module']
    assert facts_selected['interface_features']['tx_checksumming'] == 'on'
    aggregate = {d: Counter() for d in DIRECTIONS}
    rows = []
    for cid, p in catalog.items():
        assert count_orig[cid]['origin'] == int(p['origin_packets']), (cid, count_orig[cid])
        assert count_orig[cid]['responder'] == int(p['responder_packets']), (cid, count_orig[cid])
        ds = stats[cid]
        for d, c in ds.items():
            for key in ('ipv4_checksum_correct', 'ipv4_checksum_incorrect', 'ipv4_checksum_unverifiable', 'tcp_checksum_correct', 'tcp_checksum_incorrect', 'tcp_checksum_unverifiable', 'both_checksums_correct', 'ipv4_declared_bytes_checksums_correct'):
                c.setdefault(key, 0)
            c['ipv4_checksum_verifiable'] = c['ipv4_checksum_correct'] + c['ipv4_checksum_incorrect']
            assert c['packets'] == c['tcp_checksum_correct'] + c['tcp_checksum_incorrect'] + c['tcp_checksum_unverifiable']
            aggregate[d].update(c)
        matched = {}
        zeek_counts = {}
        conservation = {}
        target_key = endpoint_key(p['origin_ip'], p['origin_port'], p['responder_ip'], p['responder_port'])
        for arm, flows in arms.items():
            matched[arm] = [f for f in flows if f['native_protocol'] == 'tcp' and endpoint_key(f['source'], f['orig_port'], f['destination'], f['resp_port']) == target_key and float(p['first']) - 0.00001 <= float(f['connection_start']) <= float(p['last']) + 0.00001]
            zeek_counts[arm] = {d: Counter() for d in DIRECTIONS}
            for f in matched[arm]:
                for side in ('orig', 'resp'):
                    d = 'host_to_remote' if (f['source'] == HOST) == (side == 'orig') else 'remote_to_host'
                    for field in ('pkts', 'ip_bytes'):
                        zeek_counts[arm][d][field] += int(f[f'{side}_{field}'])
            conservation[arm] = {}
            for d in DIRECTIONS:
                raw = ds[d]
                expected_packets = raw['both_checksums_correct'] if arm == 'strict' else raw['packets']
                expected_bytes = raw['ipv4_declared_bytes_checksums_correct'] if arm == 'strict' else raw['ipv4_declared_bytes_all']
                conservation[arm][d] = {'packet_count_equal_expected': zeek_counts[arm][d]['pkts'] == expected_packets, 'ip_bytes_equal_expected': zeek_counts[arm][d]['ip_bytes'] == expected_bytes, 'expected_packets': expected_packets, 'expected_ip_bytes': expected_bytes}
                if arm == 'ignore_checksums':
                    conservation[arm][d]['packets_equal_complete_length_packets'] = zeek_counts[arm][d]['pkts'] == raw['tcp_checksum_verifiable']
                    conservation[arm][d]['ip_bytes_equal_complete_length_ip_bytes'] = zeek_counts[arm][d]['ip_bytes'] == raw['ipv4_declared_bytes_tcp_verifiable']
        h = handshakes[cid]
        h['three_way_sequence_numbers_match'] = all(h[k] is not None for k in ('syn', 'syn_ack', 'final_ack'))
        bad_lengths = any(ds[d]['tcp_checksum_unverifiable'] or ds[d]['ipv4_checksum_incorrect'] for d in DIRECTIONS)
        unexplained_bad = ds['host_to_remote']['tcp_checksum_incorrect'] != ds['host_to_remote']['tcp_incorrect_matches_pseudoheader_seed'] or ds['remote_to_host']['tcp_checksum_incorrect'] > 0
        counts_ok = all(v for arm in conservation.values() for ds1 in arm.values() for k,v in ds1.items() if k.endswith('equal_expected'))
        clear = not bad_lengths and not unexplained_bad and h['three_way_sequence_numbers_match'] and counts_ok
        rows.append({'packet_connection_id': cid, 'roles': sorted(roles[cid]), 'action_ids_or_names': actions[cid], 'packet_envelope': p, 'directions': {d:dict(c) for d,c in ds.items()}, 'handshake': h, 'zeek_rows': matched, 'zeek_directional_counts': {a:{d:dict(c) for d,c in counts.items()} for a,counts in zeek_counts.items()}, 'packet_and_byte_conservation': conservation, 'sample_bad_or_unverifiable_packets': packet_samples[cid], 'length_or_ip_integrity_problem': bad_lengths, 'unexplained_bad_tcp_checksums': unexplained_bad, 'canonical_variant': 'ignore_checksums' if clear else None, 'model_row_blocked_on_capture_semantics': not clear})
    after = {str(p.relative_to(ROOT)): checksum.sha(p) for p in frozen_paths}
    assert before == after
    result = {
        'pcap': str(pcap), 'pcap_bytes': pcap.stat().st_size, 'pcap_sha256': pcap_sha,
        'selection': {'control_requests': 32, 'control_packet_candidates': 33, 'download_actions': 2, 'download_packet_candidates': 3, 'total_packet_candidates': 36, 'control_first_request_keeps_both_candidates': [3544,3545], 'excluded_connection_ids': {'0':'Preexisting continuous 5044 log transport without SYN; historical time-window candidate, not the outbound download endpoint.'}},
        'packet_assignments': sum(c['packets'] for c in aggregate.values()), 'duplicate_packet_assignments': duplicate_assignments,
        'aggregate_by_direction': {d:dict(c) for d,c in aggregate.items()},
        'three_way_handshakes': sum(r['handshake']['three_way_sequence_numbers_match'] for r in rows),
        'capture_host_facts': facts_selected, 'capture_facts_path': str(facts_path), 'capture_facts_sha256': checksum.sha(facts_path),
        'packet_candidates': rows, 'frozen_contract_inputs_sha256': before, 'frozen_contract_unchanged': before == after,
        'archive_inputs_sha256': {name: checksum.sha(ARCHIVE/name) for name in ('tcp-connection-candidates.csv','action-connection-candidates.csv','victim-download-connection-candidates.csv')},
        'method': {'selection':'Fixed 32 historical control requests (all 33 candidates) and 3 outbound candidates of 2 downloads; no score/event-success-based flow choice.', 'packet_matching':'Exact bidirectional IPv4/TCP tuple and inclusive frozen first/last envelope; packet counts asserted separately against R27 origin/responder catalog.', 'zeek_matching':'Exact bidirectional tuple and Zeek start within fixed packet envelope, 10 microsecond numerical tolerance; all matching strict and -C records retained.', 'checksum':'RFC ones-complement checksum over IPv4 header or pseudoheader plus full IPv4-declared TCP segment. Fragments, malformed lengths, missing segment bytes are unverifiable, never counted as incorrect.', 'directions':'host_to_remote is source 10.143.2.4; independent of TCP initiator role.', 'availability':'Zeek reported end and replay export remain distinct; replay timestamps are not detector receipt or action-label availability.', 'model_clearance':'Requires host TX-offload fact, no invalid length/IP checksum, all host bad TCP checksums equal partial pseudoheader seed, no remote bad checksum, matching three-way handshake, exact both-arm packet/byte conservation.'},
        'semantic_decision': {'canonical_variant_for_cleared_candidates':'ignore_checksums', 'cleared_packet_candidate_ids':[r['packet_connection_id'] for r in rows if not r['model_row_blocked_on_capture_semantics']], 'blocked_packet_candidate_ids':[r['packet_connection_id'] for r in rows if r['model_row_blocked_on_capture_semantics']], 'strict_role':'checksum-filtered diagnostic; never merged with -C as interchangeable model rows', 'event_labels_modified':False, 'model_training_or_forward_calls':0},
        'script_sha256':checksum.sha(Path(__file__)), 'checksum_helper_sha256':checksum.sha(ROOT.parent/'ait/diagnose_checksum_semantics.py'),
        'official_sources':['https://docs.zeek.org/en/v8.0.11/quickstart.html','https://raw.githubusercontent.com/zeek/zeek/v8.0.11/scripts/base/init-bare.zeek','https://raw.githubusercontent.com/torvalds/linux/v4.15/Documentation/networking/checksum-offloads.txt','https://github.com/torvalds/linux/blob/v4.15/net/ipv4/tcp_ipv4.c#L526','https://www.rfc-editor.org/rfc/rfc9293.html#section-3.1','https://zenodo.org/records/19483937']}
    (ROOT/'checksum_semantics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cleared = [r for r in rows if not r['model_row_blocked_on_capture_semantics']]
    blocked = [r for r in rows if r['model_row_blocked_on_capture_semantics']]
    assert len(cleared) == 32 and [r['packet_connection_id'] for r in blocked] == [3544,3574,3575,3578]
    assert all(r['packet_and_byte_conservation']['ignore_checksums'][d]['packets_equal_complete_length_packets'] and r['packet_and_byte_conservation']['ignore_checksums'][d]['ip_bytes_equal_complete_length_ip_bytes'] for r in rows for d in DIRECTIONS)
    clean_counts = {d:Counter() for d in DIRECTIONS}
    for r in cleared:
        for d in DIRECTIONS:
            clean_counts[d].update(r['directions'][d])
    table = []
    for r in blocked:
        incoming = r['directions']['remote_to_host']
        crows = r['zeek_rows']['ignore_checksums']
        assert len(crows) == 1
        table.append(f"| {r['packet_connection_id']} | {incoming['packets']:,} | {incoming['tcp_checksum_unverifiable']:,} | {r['zeek_directional_counts']['ignore_checksums']['remote_to_host']['pkts']:,} | {crows[0]['missed_bytes']} |")
    md = f'''# AIT intranet 固定攻击连接的采集语义核查

32 条控制候选可采用 `ignore_checksums`（Zeek `-C`）口径。3544 及三条下载候选 3574、3575、3578 的捕获记录缺少 IPv4 声明的完整段字节，禁止进入完整 raw8 模型表。两次下载事件及其候选关系照常保留；首条控制请求仍有 3544/3545 两候选，不能把排除受损模型行当成消除了事件关联歧义。

## 固定范围与复算

读取 R27 已有 32 条控制请求的全部 33 个 TCP 候选，以及两次下载的 3574、3575、3578，共 36 条。5044 持续日志传输连接 0 不属于已锚定的外连下载，未扩入本次逐包范围。按双向端点与原始首末包时刻关联，分别断言 origin/responder 包数与 R27 目录一致；没有重复归属包。输入共 64,744 包，36 条连接全部具备序列号匹配的三次握手。

运行 `python diagnose_checksum_semantics.py` 可重建本报告及 JSON。脚本复用相邻 `ait/diagnose_checksum_semantics.py` 的 RFC 校验和函数与原 R27 PCAP 读取器，不重跑 Zeek、不改包、不改冻结 contract。全部 contract 文件执行前后 SHA256 一致。原 PCAP SHA256：`{pcap_sha}`。

## 校验和与长度分别判断

intranet 主机 facts 的快照为 `2022-01-25T06:55:29.698260Z`：`ens3=10.143.2.4`、`virtio_net`、`tx_checksumming=on`、`tx_checksum_ip_generic=on`，GRO/GSO/TSO 均为 off，Suricata 使用 ens3/af-packet。facts 来自作者 ZIP 的固定成员，CRC32 2309763221；SHA256 `{result['capture_facts_sha256']}`。快照本身不是整个采集期配置不变的证明；逐包模式提供了同方向的直接证据。

| 方向 | 包数 | IPv4 正确 | TCP 可校验 | TCP 正确 | TCP 错误 | TCP 无法校验 |
|---|---:|---:|---:|---:|---:|---:|
| intranet → 远端 | 29,226 | 29,226 | 29,226 | 2 | 29,224 | 0 |
| 远端 → intranet | 35,518 | 35,518 | 7,395 | 7,166 | 229 | 28,123 |

发送方向的 29,224 个错误 TCP checksum 全部等于 pseudoheader 部分和，符合 Linux TX checksum offload 的预填值。接收方向的 229 个可校验错误值也全部等于该部分和，但不能仅用 intranet 的 TX 配置解释远端记录。无法校验的 28,123 包全部是 IPv4 声明长度超过实际捕获字节；没有 IP 分片或零长度，PCAP 的 captured/original frame length 字段却相等，因此不能把它们写成普通 snaplen 截断，也不能据此断言网络真实丢包。[Linux 4.15 checksum offload 说明](https://raw.githubusercontent.com/torvalds/linux/v4.15/Documentation/networking/checksum-offloads.txt)、[对应 TCP 实现](https://github.com/torvalds/linux/blob/v4.15/net/ipv4/tcp_ipv4.c#L526)。

## 32 条完整控制候选

这 32 条共有 intranet 发送 {clean_counts['host_to_remote']['packets']:,} 包、远端发送 {clean_counts['remote_to_host']['packets']:,} 包。发送包全部是与 offload 部分和吻合的错误 checksum，远端包全部校验正确，长度完整。strict 恰好丢弃全部本机发送包；`-C` 的双向包数与 IP 字节数逐连接吻合原始记录。因此 `-C` 是由采集语义确定的口径，strict 只保留诊断用途，二者不可混作等价模型输入。[Zeek 8.0.11 官方说明](https://docs.zeek.org/en/v8.0.11/quickstart.html)明确说明本机 checksum offload 会影响默认校验，可用 `-C` 忽略。

这里的“可用”仅解决这 32 条固定连接的校验和与完整长度问题，不改变动作标签、首请求双候选、可用时间或同服务正常分母。完整候选 ID 与两臂所有 UID、raw8、导出时间均在 JSON。

## 四条受损候选

| packet ID | 远端原始包数 | 声明长度缺字节的包数 | -C 实际计入远端包数 | -C missed_bytes |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

这四条的 `-C` 都标成 SF，但逐包对账显示：`-C` 计入的恰好是长度完整的包，缺少完整段字节的记录全部未计入。strict 计入的则恰好是双 checksum 正确的包。两种口径都不能恢复缺失数据；`SF`、握手完成或业务成功不等于完整流特征。

例如 3574 的原始第 574025 包声明 IPv4 长度 2,761，实际仅 1,472 字节；3575 第 574045 包声明 2,759、实际 1,472；3578 第 599934 包声明 7,042、实际 1,472。JSON 保留每方向/异常类型的最早包号、时间、长度与 checksum 数值。

当前原件足以确认发送 checksum offload，也足以确认四条候选的输入缺损；尚不足以唯一确定远端/虚拟网络聚合与捕获长度处理的哪一层产生了缺损。执行决定是保留事件及全部关系，阻止四条受损完整流模型行，不用改 checksum、补长度或挑较完整候选来修饰下载结果。
'''
    (ROOT/'checksum_semantics.md').write_text(md,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('selection','packet_assignments','aggregate_by_direction','three_way_handshakes','semantic_decision','frozen_contract_unchanged')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
