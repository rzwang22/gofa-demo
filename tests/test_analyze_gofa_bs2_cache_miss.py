import unittest

from scripts.analyze_gofa_bs2_cache_miss import analyze_snapshot


class AnalyzeGofaBs2CacheMissTest(unittest.TestCase):
    def test_classifies_nog_node_and_edge_items(self):
        snapshot = {
            "batch_size_inferred": 2,
            "question_index_raw": [2, 5],
            "node_map_at_each_question_index": [
                {"question_local_index": 2, "node_map_value": 1},
                {"question_local_index": 5, "node_map_value": 4},
            ],
            "resolved_skip_cache_indices": [1, 4],
            "missing_items": [
                {
                    "cache_key": "nog-key",
                    "item_index": 4,
                    "item_type": "node",
                    "is_question_or_nog_candidate": True,
                },
                {
                    "cache_key": "node-key",
                    "item_index": 0,
                    "item_type": "node",
                    "is_question_or_nog_candidate": False,
                },
                {
                    "cache_key": "edge-key",
                    "item_index": 6,
                    "item_type": "edge",
                    "is_question_or_nog_candidate": False,
                },
                {
                    "cache_key": "unknown-key",
                    "item_index": 7,
                    "item_type": "unknown",
                    "is_question_or_nog_candidate": False,
                },
            ],
        }

        result = analyze_snapshot(snapshot)

        self.assertEqual(result["number_of_question_indices"], 2)
        self.assertEqual(result["number_of_distinct_mapped_question_items"], 2)
        self.assertEqual(result["number_of_skipped_items"], 2)
        self.assertTrue(result["counts_consistent"])
        self.assertTrue(result["all_mapped_question_items_skipped"])
        self.assertEqual(
            [item["classification"] for item in result["missing_items"]],
            ["NOG candidate", "node text item", "edge text item", "unresolved"],
        )


if __name__ == "__main__":
    unittest.main()
