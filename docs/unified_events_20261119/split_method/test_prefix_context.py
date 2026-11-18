"""Check availability and membership contracts, without fitting a model."""
import itertools
import math
import unittest

from prefix_context import build_prefix_context


def flow(flow_id, when, value=0, *, group="capture-a", host="monitored-a", peer="peer-a", **extra):
    return dict(group=group, host=host, peer=peer, available_at=when,
                flow_id=flow_id, raw8=[value] * 8, **extra)


def by_id(result):
    return {r["flow_id"]: r for r in result["rows"]}


class PrefixContextTests(unittest.TestCase):
    def test_equal_time_batch_cannot_see_itself_and_tie_break_only_affects_later_state(self):
        records = [flow("b", 10, 3), flow("a", 10, 1), flow("c", 12, 8)]
        result = build_prefix_context(records)
        rows = by_id(result)
        for fid in ("a", "b"):
            self.assertEqual(rows[fid]["history_features"], [0.0] * 18 + [1])
            self.assertIsNone(rows[fid]["max_history_available_at"])
        later = rows["c"]
        self.assertEqual(later["previous_flow_id"], "b")
        self.assertEqual(later["history_flow_count"], 2)
        self.assertAlmostEqual(later["prefix_mean_log1p"][0], (math.log(2) + math.log(4)) / 2)
        self.assertAlmostEqual(later["log1p_elapsed_since_previous_available"], math.log(3))
        self.assertEqual(later["max_history_available_at"], 10)

    def test_future_records_do_not_change_earlier_context(self):
        past = [flow("a", 1, 2), flow("b", 2, 5)]
        before = by_id(build_prefix_context(past))
        after = by_id(build_prefix_context([flow("future", 100, 999)] + past))
        self.assertEqual(before["a"], after["a"])
        self.assertEqual(before["b"], after["b"])
        self.assertTrue(build_prefix_context(past)["audit"]["strictly_past_history"])

    def test_groups_and_monitored_hosts_have_separate_state(self):
        result = by_id(build_prefix_context([
            flow("a1", 1, 3), flow("a2", 2, 6),
            flow("other-host", 3, 9, host="monitored-b"),
            flow("b1", 4, 50, group="capture-b"),
            flow("b2", 5, 60, group="capture-b"),
        ]))
        self.assertEqual(result["other-host"]["empty_history"], 1)
        self.assertEqual(result["b1"]["empty_history"], 1)
        self.assertEqual(result["b2"]["previous_flow_id"], "b1")
        self.assertEqual(result["b2"]["history_flow_count"], 1)

    def test_unlabelled_background_is_consumed_and_labels_cannot_change_context(self):
        records = [flow("attack", 1, 1, label="attack"),
                   flow("unlabelled-background", 2, 9, peer="ordinary-peer"),
                   flow("current", 3, 4, peer="ordinary-peer", label="normal")]
        original = build_prefix_context(records)
        current = by_id(original)["current"]
        self.assertEqual(current["previous_flow_id"], "unlabelled-background")
        self.assertEqual(current["history_flow_count"], 2)
        self.assertEqual(current["same_peer_as_previous"], 1)
        self.assertEqual(original["audit"]["history_updates"], 3)
        relabelled = [{**r, "label": "unknown", "action_id": "irrelevant"} for r in records]
        self.assertEqual(original, build_prefix_context(relabelled))

    def test_all_input_permutations_produce_identical_rows_and_means(self):
        records = [flow("b", 2, 1), flow("a", 2, 100), flow("c", 1, 0), flow("d", 5, 4)]
        expected = build_prefix_context(records)
        for permutation in itertools.permutations(records):
            self.assertEqual(build_prefix_context(permutation), expected)

    def test_missing_available_time_requires_explicit_offline_clock_for_every_record(self):
        records = [flow("a", None, 2, flow_end_at=10),
                   flow("b", 500, 5, flow_end_at=12)]
        with self.assertRaisesRegex(ValueError, "available_at"):
            build_prefix_context(records)
        result = build_prefix_context(records, time_mode="offline_flow_end_replay")
        rows = by_id(result)
        self.assertEqual(rows["b"]["effective_available_at"], 12)
        self.assertEqual(rows["b"]["max_history_available_at"], 10)
        self.assertFalse(result["actual_online_availability_claimed"])
        self.assertIn("not actual detector", result["clock_semantics"])
        with self.assertRaisesRegex(ValueError, "flow_end_at"):
            build_prefix_context([flow("x", 30)], time_mode="offline_flow_end_replay")

    def test_equivalent_utc_offsets_are_one_batch_and_naive_times_are_rejected(self):
        rows = by_id(build_prefix_context([
            flow("a", "2019-09-25T10:00:00-04:00"),
            flow("b", "2019-09-25T14:00:00Z"),
            flow("c", "2019-09-25T14:00:01+00:00"),
        ]))
        self.assertEqual(rows["a"]["effective_available_at"], rows["b"]["effective_available_at"])
        self.assertEqual(rows["b"]["empty_history"], 1)
        self.assertEqual(rows["c"]["history_flow_count"], 2)
        with self.assertRaisesRegex(ValueError, "explicit timezone"):
            build_prefix_context([flow("naive", "2019-09-25T14:00:00")])

    def test_block_group_keeps_complete_other_group_and_audits_every_exclusion(self):
        normal = [flow("normal-1", None, 3, group="normal", flow_end_at=1),
                  flow("normal-2", None, 7, group="normal", flow_end_at=2)]
        bad = [flow("empire-earlier", None, 5, group="empire", flow_end_at=1),
               flow("empire-missing", None, 9, group="empire", flow_end_at=None, provenance="original marker"),
               flow("empire-later", None, 10, group="empire", flow_end_at=3)]
        with self.assertRaisesRegex(ValueError, "flow_end_at"):
            build_prefix_context(normal + bad, time_mode="offline_flow_end_replay")
        result = build_prefix_context(normal + bad, time_mode="offline_flow_end_replay", unavailable_policy="block-group")
        self.assertEqual(result["rows"], build_prefix_context(normal, time_mode="offline_flow_end_replay")["rows"])
        audit = result["audit"]
        self.assertEqual((audit["input_observations"], audit["output_observations"], audit["blocked_observations"]), (5, 2, 3))
        self.assertEqual(audit["history_updates"], 2)
        self.assertTrue(audit["all_supplied_rows_accounted_for"])
        self.assertEqual(audit["blocked_groups"][0]["flow_ids"], ["empire-earlier", "empire-later", "empire-missing"])
        self.assertEqual(audit["unavailable_observations"][0]["original_observation"], bad[1])
        self.assertEqual(audit["unavailable_observation_count"], 1)
        self.assertEqual(result, build_prefix_context(reversed(normal + bad),
            time_mode="offline_flow_end_replay", unavailable_policy="block-group"))

    def test_block_group_does_not_turn_invalid_nonnull_clock_into_missing(self):
        with self.assertRaisesRegex(ValueError, "Invalid flow_end_at"):
            build_prefix_context([flow("bad", None, flow_end_at="not-a-time")],
                time_mode="offline_flow_end_replay", unavailable_policy="block-group")

    def test_duplicate_flow_ids_are_rejected_even_across_groups(self):
        for second in (flow("same", 2, 8), flow("same", 2, 8, group="other")):
            with self.assertRaisesRegex(ValueError, "Duplicate flow_id"):
                build_prefix_context([flow("same", 1), second])

    def test_two_host_views_share_one_physical_flow_without_double_counting(self):
        views = [flow("view-a", 1, physical_flow_id="physical-1"),
                 flow("view-b", 1, host="monitored-b", physical_flow_id="physical-1")]
        result = build_prefix_context(views)
        self.assertEqual(result["audit"]["host_views"], 2)
        self.assertEqual(result["audit"]["unique_physical_flows"], 1)
        self.assertTrue(all(r["empty_history"] == 1 for r in result["rows"]))
        with self.assertRaisesRegex(ValueError, "same monitored host"):
            build_prefix_context(views + [flow("duplicate-view", 2, physical_flow_id="physical-1")])
        with self.assertRaisesRegex(ValueError, "cross capture groups"):
            build_prefix_context(views + [flow("other-group-view", 2, group="capture-b",
                host="another-host", physical_flow_id="physical-1")])

    def test_empty_history_zero_is_distinct_from_existing_missing_measurements(self):
        first = flow("missing", 1)
        first["raw8"] = [None, 3] + [0] * 6
        rows = by_id(build_prefix_context([first, flow("second", 2)]))
        self.assertEqual(rows["missing"]["history_features"], [0.0] * 18 + [1])
        self.assertEqual(rows["second"]["empty_history"], 0)
        self.assertIsNone(rows["second"]["previous_raw8_log1p"][0])
        self.assertIsNone(rows["second"]["prefix_mean_log1p"][0])
        self.assertEqual(rows["second"]["mean_valid_counts"][:2], [0, 1])
        self.assertEqual(len(rows["second"]["history_features"]), 19)
        self.assertEqual(len(rows["second"]["model_features"]), 27)


if __name__ == "__main__":
    unittest.main()
