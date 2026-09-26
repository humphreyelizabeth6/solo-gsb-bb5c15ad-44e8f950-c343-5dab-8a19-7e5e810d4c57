"""API integration tests (require fastapi + httpx).

Run with either:
    python3 -m unittest discover -s tests
    pytest
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.srt import format_timestamp as _ts


def _word(text, start_ms, end_ms):
    return {"text": text, "start_ms": start_ms, "end_ms": end_ms}


BODY = {
    "words": [
        _word("Hello", 0, 500),
        _word("world.", 500, 1400),
        _word("How", 2000, 2400),
        _word("are", 2400, 2700),
        _word("you", 2700, 3200),
        _word("doing", 3200, 3800),
        _word("today?", 3800, 4600),
        _word("Fine", 5200, 5600),
        _word("thanks.", 5600, 6400),
    ],
    "max_chars": 42,
    "min_duration_ms": 1000,
    "max_duration_ms": 7000,
    "target_duration_ms": 3000,
    "pause_threshold_ms": 500,
}


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_segment_json(self):
        r = self.client.post("/segment", json=BODY)
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["segment_count"], 3)
        segs = data["segments"]
        self.assertEqual(segs[0]["start_ms"], 0)
        self.assertEqual(segs[-1]["end_ms"], 6400)
        # natural boundaries (punctuation or >=500 ms gaps) should land on
        # segment edges, not be crossed
        self.assertEqual(sum(s["crossed_boundaries"] for s in segs), 0)
        # every word covered once, in order
        self.assertEqual(segs[0]["word_start"], 0)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(b["word_start"], a["word_end"] + 1)
        self.assertEqual(segs[-1]["word_end"], len(BODY["words"]) - 1)
        for s in segs:
            self.assertLessEqual(len(s["text"]), 42)
            self.assertGreaterEqual(s["duration_ms"], 1000)
            self.assertLessEqual(s["duration_ms"], 7000)

    def test_segment_srt(self):
        r = self.client.post("/segment/srt", json=BODY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        srt = r.text
        self.assertTrue(srt.startswith("1\n"))
        self.assertIn("00:00:00,000 --> 00:00:01,400", srt)
        self.assertIn("Hello world.", srt)
        self.assertIn("Fine thanks.", srt)

    def test_422_on_overlong_word(self):
        body = {**BODY, "words": [_word("x" * 50, 0, 500)]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("max_chars", r.json()["detail"])

    def test_422_on_overlapping_words(self):
        body = {
            **BODY,
            "words": [_word("a", 0, 500), _word("b", 400, 900)],
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("non-overlapping", r.json()["detail"])

    def test_422_on_impossible_min_duration(self):
        body = {
            **BODY,
            "words": [_word("a", 0, 300)],
            "min_duration_ms": 1000,
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("no legal segmentation", r.json()["detail"])

    def test_422_on_schema_violation(self):
        body = {**BODY, "words": [_word("a", 500, 100)]}  # end < start
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_defaults_applied_when_parameters_omitted(self):
        r = self.client.post(
            "/segment",
            json={"words": BODY["words"]},
        )
        self.assertEqual(r.status_code, 200, r.text)

    def test_max_line_chars_lays_text_out_on_two_lines(self):
        words = [
            _word("aaa", 0, 450),
            _word("bbb", 450, 900),
            _word("ccc", 900, 1350),
            _word("ddd", 1350, 1800),
        ]
        body = {
            "words": words,
            "max_chars": 42,
            "min_duration_ms": 0,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 7,
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["text"], "aaa bbb\nccc ddd")
        for line in segs[0]["text"].split("\n"):
            self.assertLessEqual(len(line), 7)

    def test_max_line_chars_srt_uses_newline(self):
        words = [
            _word("aaa", 0, 450),
            _word("bbb", 450, 900),
            _word("ccc", 900, 1350),
            _word("ddd", 1350, 1800),
        ]
        body = {
            "words": words,
            "max_chars": 42,
            "min_duration_ms": 0,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 7,
        }
        r = self.client.post("/segment/srt", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("aaa bbb\nccc ddd\n", r.text)

    def test_422_when_line_width_makes_segmentation_impossible(self):
        words = [
            _word("aaaa", 0, 400),
            _word("bbbb", 400, 800),
            _word("cccc", 800, 1200),
        ]
        body = {
            "words": words,
            "max_chars": 42,
            "min_duration_ms": 1000,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 4,
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("no legal segmentation", r.json()["detail"])

    def test_max_line_chars_must_be_positive(self):
        body = {**BODY, "max_line_chars": 0}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    # ---- manual editing boundaries ----------------------------------------
    def test_required_break_after_forces_boundary(self):
        body = {**BODY, "required_break_after": [4]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        edges = [s["word_end"] for s in segs[:-1]]
        self.assertIn(4, edges)
        # every word still covered exactly once, in order
        self.assertEqual(segs[0]["word_start"], 0)
        self.assertEqual(segs[-1]["word_end"], len(BODY["words"]) - 1)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(b["word_start"], a["word_end"] + 1)

    def test_keep_together_keeps_words_in_one_subtitle(self):
        body = {**BODY, "keep_together": [[3, 5]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        edges = [s["word_end"] for s in segs[:-1]]
        for inner in (3, 4):  # inner edges of [3, 5]
            self.assertNotIn(inner, edges)
        owner = next(s for s in segs if s["word_start"] <= 3 <= s["word_end"])
        self.assertGreaterEqual(owner["word_end"], 5)

    def test_boundaries_applied_identically_to_srt(self):
        body = {
            **BODY,
            "min_duration_ms": 0,
            "required_break_after": [4],
            "keep_together": [[1, 2]],
        }
        rj = self.client.post("/segment", json=body)
        rs = self.client.post("/segment/srt", json=body)
        self.assertEqual(rj.status_code, 200, rj.text)
        self.assertEqual(rs.status_code, 200, rs.text)
        # both kinds of boundary are honoured ...
        segs = rj.json()["segments"]
        edges = [s["word_end"] for s in segs[:-1]]
        self.assertIn(4, edges)
        self.assertNotIn(1, edges)
        # ... and the SRT blocks describe exactly the JSON segments
        blocks = rs.text.strip().split("\n\n")
        self.assertEqual(len(blocks), len(segs))
        for n, block in enumerate(blocks, start=1):
            lines = block.split("\n")
            self.assertEqual(lines[0], str(n))
            seg = segs[n - 1]
            self.assertEqual(
                lines[1],
                f"{_ts(seg['start_ms'])} --> {_ts(seg['end_ms'])}",
            )
            self.assertEqual("\n".join(lines[2:]), seg["text"])

    def test_boundaries_combine_with_two_line_layout(self):
        words = [
            _word("aaa", 0, 450),
            _word("bbb", 450, 900),
            _word("ccc", 900, 1350),
            _word("ddd", 1350, 1800),
        ]
        body = {
            "words": words,
            "max_chars": 42,
            "min_duration_ms": 0,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 7,
            "required_break_after": [1],
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        self.assertEqual([s["word_end"] for s in segs[:-1]], [1])
        for s in segs:
            for line in s["text"].split("\n"):
                self.assertLessEqual(len(line), 7)

    def test_422_on_break_inside_keep_together(self):
        body = {**BODY, "required_break_after": [4], "keep_together": [[3, 5]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("falls inside keep_together", r.json()["detail"])

    def test_422_on_required_break_out_of_range(self):
        body = {**BODY, "required_break_after": [len(BODY["words"]) - 1]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("out of range", r.json()["detail"])

    def test_422_on_required_break_negative(self):
        body = {**BODY, "required_break_after": [-1]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_422_on_keep_together_too_short(self):
        body = {**BODY, "keep_together": [[2, 2]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("at least two words", r.json()["detail"])

    def test_422_on_overlapping_keep_together(self):
        body = {**BODY, "keep_together": [[0, 3], [3, 5]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("overlap", r.json()["detail"])

    def test_422_on_keep_together_out_of_range(self):
        body = {**BODY, "keep_together": [[8, 9]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_422_when_boundaries_make_segmentation_impossible(self):
        # the forced first subtitle would contain only one 500 ms word,
        # below the 1000 ms minimum duration
        body = {**BODY, "required_break_after": [0]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("no legal segmentation", r.json()["detail"])

    def test_422_on_malformed_interval_shape(self):
        body = {**BODY, "keep_together": [[1, 2, 3]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_empty_boundary_lists_are_accepted(self):
        body = {
            **BODY,
            "required_break_after": [],
            "keep_together": [],
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)


if __name__ == "__main__":
    unittest.main()
