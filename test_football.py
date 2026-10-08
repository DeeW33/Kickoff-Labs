"""Automated checks for the parts of the system that handle money-like bookkeeping: grading, the top-5 rule, duplicate
handling, the pick log's safety net and the Opta freshness rules. Run with:  python -m unittest discover -s tests -v
No network access is needed."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "..")):
    sys.path.insert(0, _p)
import football_predictor as fp  # noqa: E402


def load(path):
    with open(path) as fh:
        return json.load(fh)


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def make_games(season=2026, weeks=(1,), per_week=8, played=True):
    rows = []
    for w in weeks:
        for k in range(per_week):
            rows.append({"game_id": f"{season}_{w:02d}_A{k}_H{k}", "season": season, "week": w,
                         "date": pd.Timestamp(f"{season}-09-{7 + 7 * (w - 1):02d}"), "gametime": "13:00",
                         "home": f"H{k}", "away": f"A{k}", "neutral": False,
                         "home_pts": 24.0 if played else np.nan, "away_pts": 17.0 if played else np.nan})
    return pd.DataFrame(rows)


def make_allr(games, margins):
    """Model predictions per game: pred_margin chosen so that edge vs a flat -0 line equals `margins`."""
    d = pd.DataFrame(index=games.index)
    d["season"] = games.season
    d["pred_margin"] = margins
    d["pred_total"] = 45.0
    d["pred_home"] = (45.0 + d.pred_margin) / 2
    d["pred_away"] = (45.0 - d.pred_margin) / 2
    d["mkt_spread"] = -3.5          # home favored by 3.5 (not a pick'em, so these are spread plays)
    d["mkt_total"] = 45.0
    d["mkt_ml_home"] = np.nan
    d["mkt_ml_away"] = np.nan
    d["actual_margin"] = games.home_pts - games.away_pts
    d["actual_total"] = games.home_pts + games.away_pts
    return d


def entry(gid, **kw):
    e = {"id": f"{gid}|spread", "type": "spread", "ml": None, "game_id": gid, "league": "nfl", "season": 2026, "week": 1,
         "kickoff": "2026-09-07T17:00:00Z", "away": "A", "home": "H", "team": "H", "side": None, "line": -3.5,
         "edge": 4.0, "units": 1.5, "status": "graded", "home_pts": 24.0, "away_pts": 17.0, "result": 1, "net": 1.36,
         "logged_at": "2026-09-05T00:00:00Z"}
    e.update(kw)
    return e


class Grading(unittest.TestCase):
    def test_spread_win_loss_push(self):
        e = entry("g", team="H", line=-3.5, units=1.0)
        self.assertEqual(fp.grade_pick(e, 24, 17)[0], 1)                 # wins by 7, covers -3.5
        self.assertEqual(fp.grade_pick(e, 20, 17)[0], -1)                # wins by 3, does not cover
        self.assertEqual(fp.grade_pick(entry("g", team="H", line=-3.0), 20, 17)[0], 0)   # exactly 3: push
        self.assertEqual(fp.grade_pick(entry("g", team="H", line=-3.0), 20, 17)[1], 0.0)

    def test_underdog_side(self):
        e = entry("g", team="A", line=3.5)                                # away +3.5
        self.assertEqual(fp.grade_pick(e, 24, 21)[0], 1)                  # loses by 3, covers
        self.assertEqual(fp.grade_pick(e, 24, 17)[0], -1)

    def test_units_and_payout(self):
        res, net = fp.grade_pick(entry("g", units=1.5, line=-3.5), 24, 17)
        self.assertAlmostEqual(net, 1.5 * fp.WIN_UNITS)
        res, net = fp.grade_pick(entry("g", units=1.5, line=-3.5), 20, 17)
        self.assertAlmostEqual(net, -1.5)

    def test_total(self):
        over = entry("g", type="total", side="Over", line=40.5, team=None)
        self.assertEqual(fp.grade_pick(over, 24, 17)[0], 1)               # 41 > 40.5
        self.assertEqual(fp.grade_pick(entry("g", type="total", side="Under", line=40.5, team=None), 24, 17)[0], -1)
        self.assertEqual(fp.grade_pick(entry("g", type="total", side="Over", line=41.0, team=None), 24, 17)[0], 0)

    def test_moneyline(self):
        e = entry("g", type="ml", team="H", ml=-120.0, line=None, units=1.0)
        self.assertEqual(fp.grade_pick(e, 24, 17)[0], 1)
        self.assertEqual(fp.grade_pick(e, 17, 24)[0], -1)


class PickRules(unittest.TestCase):
    def test_threshold_and_units(self):
        small = fp.make_picks("H", "A", -1.0, 45, -3.5, 45)               # model -1.0 vs market -3.5: edge 2.5 -> no play
        self.assertIsNone(small["spread_play"])
        self.assertIsNotNone(small["sim_spread"])                         # but the model's side is still shown
        mid = fp.make_picks("H", "A", 0.0, 45, -3.5, 45)["spread_play"]   # edge 3.5
        self.assertEqual(mid["units"], 1.0)
        big = fp.make_picks("H", "A", 0.5, 45, -4.5, 45)["spread_play"]   # edge 5 -> strong
        self.assertEqual(big["units"], 1.5)

    def test_pickem_becomes_moneyline(self):
        p = fp.make_picks("H", "A", 4.0, 45, -1.0, 45, -115, -105)["spread_play"]   # spread within 2 -> moneyline
        self.assertEqual(p.get("kind"), "ml")


class Duplicates(unittest.TestCase):
    def test_old_entries_without_season_week(self):
        games = make_games(weeks=(4,), per_week=2)
        gid = games.game_id.iloc[0]
        old = entry(gid); old.pop("season"); old.pop("week")              # an old entry that lacks both fields
        new = entry(gid, replay=True, rebuilt=2, logged_at="2026-10-07T00:00:00Z")
        out = fp.dedupe_log([old, new], games)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].get("replay"))                            # the original live entry wins
        self.assertEqual((out[0]["season"], out[0]["week"]), (2026, 4))

    def test_same_game_different_kind_kept(self):
        games = make_games(weeks=(1,), per_week=1)
        gid = games.game_id.iloc[0]
        out = fp.dedupe_log([entry(gid), entry(gid, id=f"{gid}|total", type="total", side="Over", team=None, line=44.5)], games)
        self.assertEqual(len(out), 2)

    def test_graded_beats_pending(self):
        games = make_games(weeks=(1,), per_week=1)
        gid = games.game_id.iloc[0]
        out = fp.dedupe_log([entry(gid, status="pending"), entry(gid)], games)
        self.assertEqual([e["status"] for e in out], ["graded"])


class Top5(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "picks_nfl.json")
        fp.HEALTH_STATS.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def cands(self, games, edges):
        out = []
        for (i, g), edge in zip(games.iterrows(), edges):
            pk = fp.make_picks(g.home, g.away, -3.5 + edge, 45.0, -3.5, 45.0)   # model margin = line + edge on the home side
            out.append({"game_id": g.game_id, "season": 2026, "week": int(g.week), "kickoff": "2026-09-13T17:00:00Z",
                        "away": g.away, "home": g.home, "pred_away": 20, "pred_home": 24, "spread_play": pk["spread_play"],
                        "total_play": pk["total_play"], "sim_spread": pk["sim_spread"], "mkt_spread": -3.5, "mkt_total": 45.0,
                        "mkt_ml_home": None, "mkt_ml_away": None})
        return out

    def run_log(self, games, edges):
        now = pd.Timestamp("2026-09-11T12:00:00Z")
        live, official = fp.update_pick_log(self.path, "nfl", self.cands(games, edges), games, now)
        return load(self.path), official

    def test_only_five_even_with_many_qualifying(self):
        games = make_games(per_week=8, played=False)
        log, _ = self.run_log(games, [6, 5.5, 5, 4.5, 4, 3.5, 3.2, 3.1])
        self.assertEqual(len(log), 5)
        self.assertEqual(sorted(e["edge"] for e in log), sorted([6, 5.5, 5, 4.5, 4]))

    def test_no_filler_when_fewer_than_five_qualify(self):
        games = make_games(per_week=8, played=False)
        log, _ = self.run_log(games, [6, 4.2, 3.4, 2.9, 2.0, 1.0, 0.5, 0.1])
        self.assertEqual(len(log), 3)
        self.assertTrue(all(e["edge"] >= fp.EDGE_PLAY for e in log))

    def test_totals_never_official(self):
        games = make_games(per_week=6, played=False)
        c = self.cands(games, [4, 4, 4, 4, 4, 4])
        for u in c:
            u["total_play"] = {"side": "Over", "line": 40.0, "edge": 9.0, "units": 1.5}   # huge total edges exist
        now = pd.Timestamp("2026-09-11T12:00:00Z")
        fp.update_pick_log(self.path, "nfl", c, games, now)
        self.assertTrue(all(e["type"] != "total" for e in load(self.path)))

    def test_backfill_removes_totals_and_fillers_and_is_stable(self):
        games = make_games(weeks=(1,), per_week=8, played=True)
        allr = make_allr(games, [0.0, 0.5, 1.0, 2.5, 3.0, 3.5, 4.0, 4.5])      # edges 3.5,3.0,2.5,... vs the -3.5 line
        total = entry("x", id="x|total", type="total", side="Over", team=None, line=44.5, game_id=games.game_id.iloc[0])
        filler = entry(games.game_id.iloc[3], edge=1.5, units=1.0, rebuilt=2, replay=True)
        dump([total, filler], self.path)
        now = pd.Timestamp("2026-10-07T00:00:00Z")
        fp.spread_only_backfill(self.path, allr, games, 2026, now)
        log = load(self.path)
        self.assertTrue(all(e["type"] != "total" for e in log))
        self.assertTrue(all(e["edge"] >= fp.EDGE_PLAY - 1e-9 for e in log))
        self.assertLessEqual(len(log), fp.TOP_N)
        ids = [e["game_id"] for e in log]
        self.assertEqual(len(ids), len(set(ids)))
        before = load(self.path)
        fp.spread_only_backfill(self.path, allr, games, 2026, now)                # second run changes nothing
        self.assertEqual(before, load(self.path))


class LogSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "picks_nfl.json")
        fp.HEALTH_STATS.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_is_empty(self):
        self.assertEqual(fp.load_log_file(self.path), [])

    def test_corrupt_json_stops_the_run(self):
        with open(self.path, "w") as fh:
            fh.write("{not json")
        with self.assertRaises(SystemExit):
            fp.load_log_file(self.path)
        with open(self.path) as fh:
            self.assertEqual(fh.read(), "{not json")                 # and it was not overwritten

    def test_wrong_shape_stops_the_run(self):
        dump({"picks": []}, self.path)
        with self.assertRaises(SystemExit):
            fp.load_log_file(self.path)

    def test_malformed_entries_set_aside(self):
        good = entry("g1")
        dump([good, {"id": "broken"}, "junk"], self.path)
        self.assertEqual(fp.load_log_file(self.path), [good])
        self.assertTrue(os.path.exists(self.path.replace(".json", "_invalid.json")))
        self.assertEqual(fp.HEALTH_STATS["log"]["invalid"], 2)

    def test_backup_written_and_restore_after_old_upload(self):
        full = [entry("g1"), entry("g2"), entry("g3")]
        fp.write_log_file(self.path, full)                                    # a normal run: log + dated backup
        backups = os.listdir(os.path.join(self.tmp.name, "backups"))
        self.assertEqual(len(backups), 1)
        dump(full[:1], self.path)                             # someone uploads an old, shorter file
        restored = fp.restore_from_backup(self.path, fp.load_log_file(self.path))
        self.assertEqual({e["game_id"] for e in restored}, {"g1", "g2", "g3"})
        self.assertEqual(fp.HEALTH_STATS["log"]["restored"], 2)

    def test_no_restore_when_nothing_missing(self):
        full = [entry("g1"), entry("g2")]
        fp.write_log_file(self.path, full)
        out = fp.restore_from_backup(self.path, fp.load_log_file(self.path))
        self.assertEqual(len(out), 2)
        self.assertNotIn("restored", fp.HEALTH_STATS.get("log", {}))


class OptaFreshness(unittest.TestCase):
    NOW = pd.Timestamp("2026-10-08T12:00:00Z")

    def feed(self, updated):
        j = {"offense": [{"player": "Tristan Wirfs", "positionGroup": "OL", "teamAbbreviation": "TB",
                          "passBlockPctRank": 98, "runBlockPctRank": 99, "snaps": 239}], "defense": [], "qb": []}
        if updated:
            j["lastUpdated"] = updated
        return j

    def run_load(self, file_json, live_ok=False, live_json=None):
        fp.HEALTH_STATS.clear()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "opta.json")
            if file_json is not None:
                dump(file_json, p)
            resp = mock.Mock()
            resp.raise_for_status = lambda: None
            resp.json = lambda: live_json
            with mock.patch.object(fp, "OPTA_FILE", p), \
                    mock.patch.object(fp.requests, "get", side_effect=(lambda *a, **k: resp) if live_ok else Exception("blocked")):
                return fp.load_opta(now=self.NOW)

    def test_fresh_file_used(self):
        idx = self.run_load(self.feed("2026-10-05T00:00:00Z"))
        self.assertEqual(len(idx), 1)
        self.assertAlmostEqual(fp.HEALTH_STATS["opta"]["age_days"], 3.5, places=1)

    def test_stale_file_ignored(self):
        idx = self.run_load(self.feed("2026-09-20T00:00:00Z"))
        self.assertEqual(idx, {})
        self.assertTrue(fp.HEALTH_STATS["opta"]["ignored"])

    def test_undated_file_used_with_note(self):
        idx = self.run_load(self.feed(None))
        self.assertEqual(len(idx), 1)
        self.assertIn("no lastUpdated", fp.HEALTH_STATS["opta"]["note"])

    def test_live_preferred_when_fresh(self):
        idx = self.run_load(None, live_ok=True, live_json=self.feed("2026-10-08T10:00:00Z"))
        self.assertEqual(fp.HEALTH_STATS["opta"]["source"], "live")
        self.assertEqual(len(idx), 1)

    def test_stale_live_falls_back_to_fresh_file(self):
        idx = self.run_load(self.feed("2026-10-07T00:00:00Z"), live_ok=True, live_json=self.feed("2026-08-01T00:00:00Z"))
        self.assertEqual(fp.HEALTH_STATS["opta"]["source"], "file")
        self.assertEqual(len(idx), 1)

    def test_nothing_available(self):
        self.assertEqual(self.run_load(None), {})
        self.assertEqual(fp.HEALTH_STATS["opta"]["source"], "none")


class Health(unittest.TestCase):
    NOW = pd.Timestamp("2026-10-08T12:00:00Z")

    def upcoming(self, n=10, with_lines=True):
        return [{"vegas_spread": -3.0 if with_lines else None, "pred_margin": 2.0, "pred_total": 45.0} for _ in range(n)]

    def test_all_good(self):
        fp.HEALTH_STATS.clear()
        games = make_games(per_week=2, played=True)
        games["date"] = pd.Timestamp("2026-10-05")
        h = fp.build_health(self.upcoming(), games, self.NOW)
        self.assertIn(h["level"], ("ok", "info"))

    def test_missing_lines_is_error(self):
        fp.HEALTH_STATS.clear()
        h = fp.build_health(self.upcoming(with_lines=False), make_games(played=True), self.NOW)
        self.assertEqual(h["level"], "error")

    def test_absurd_prediction_is_error(self):
        fp.HEALTH_STATS.clear()
        up = self.upcoming(); up[0]["pred_margin"] = 55.0
        self.assertEqual(fp.build_health(up, make_games(played=True), self.NOW)["level"], "error")

    def test_stale_opta_warns(self):
        fp.HEALTH_STATS.clear()
        fp.HEALTH_STATS["opta"] = {"source": "file", "ignored": True, "players": 0, "note": "file: ratings are 14 days old"}
        games = make_games(played=True); games["date"] = pd.Timestamp("2026-10-05")
        h = fp.build_health(self.upcoming(), games, self.NOW)
        self.assertEqual(h["level"], "warn")
        self.assertTrue(any("Opta" in m for m in h["messages"]))


if __name__ == "__main__":
    unittest.main()
