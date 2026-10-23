import json
from pathlib import Path

rows = json.loads(
    Path(r"C:\Users\RobDods\Apps\Cursor\Predictors\Cursor_NHLPO_Predict\data\processed\yahoo_jeffs_draft_board.json").read_text(encoding="utf-8")
)


def show(title, xs, n=40):
    print("====", title, "n=", len(xs))
    for i, r in enumerate(xs[:n], 1):
        print(
            f"{i:2d} ovr={r['rank_overall']:3d} {r['player']:22s} {r['team']:3s} "
            f"{r['yahoo_pos']:4s} {r['nhl_pos']:3s} {r['proj_fp']:6.1f} gp={r['exp_gp']:2d} {r['flag']}"
        )


show("OVR", rows, 40)
show("F", [r for r in rows if r["f"]], 40)
show("C", [r for r in rows if r["c"]], 15)
show("LW", [r for r in rows if r["lw"]], 12)
show("RW", [r for r in rows if r["rw"]], 12)
show("D", [r for r in rows if r["d"]], 15)
show("G", [r for r in rows if r["g"]], 20)
print("avoid", [(r["rank_overall"], r["player"], r["flag"]) for r in rows if r["avoid"]])

# compact canvas objects
keys = ["rank_overall", "player", "team", "yahoo_pos", "nhl_pos", "proj_fp", "last_year_fp", "last_year_gp", "our_nhl_pred_if_any", "note", "flag", "avoid", "c", "lw", "rw", "f", "d", "g", "goal"]
out = [{k: r.get(k) for k in keys} for r in rows]
Path(r"C:\Users\RobDods\Apps\Cursor\Predictors\Cursor_NHLPO_Predict\data\processed\_canvas_players.json").write_text(
    json.dumps(out), encoding="utf-8"
)
print("canvas n", len(out))
