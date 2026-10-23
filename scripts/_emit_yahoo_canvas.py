"""Emit yahoo-jeffs-draft-board.canvas.tsx from the board JSON."""
from __future__ import annotations

import json
from pathlib import Path

JSON_PATH = Path(
    r"C:\Users\RobDods\Apps\Cursor\Predictors\Cursor_NHLPO_Predict\data\processed\yahoo_jeffs_draft_board.json"
)
CANVAS = Path(
    r"C:\Users\RobDods\.cursor\projects\c-Users-RobDods-Apps-Cursor-Predictors-Cursor-NHLPO-Predict\canvases\yahoo-jeffs-draft-board.canvas.tsx"
)

rows = json.loads(JSON_PATH.read_text(encoding="utf-8"))


def js_str(s) -> str:
    if s is None:
        return "null"
    return json.dumps(str(s), ensure_ascii=False)


def num_or_null(v):
    if v in ("", None):
        return "null"
    return str(int(v))


lines = []
for r in rows:
    last = r.get("last_year_fp") or 0
    our = r.get("our_nhl_pred_if_any")
    our_js = "null" if our in ("", None) else str(float(our))
    lines.append(
        "{"
        f"rk:{r['rank_overall']},"
        f"rf:{num_or_null(r.get('rank_f'))},"
        f"rc:{num_or_null(r.get('rank_c'))},"
        f"rl:{num_or_null(r.get('rank_lw'))},"
        f"rr:{num_or_null(r.get('rank_rw'))},"
        f"rd:{num_or_null(r.get('rank_d'))},"
        f"rg:{num_or_null(r.get('rank_g'))},"
        f"name:{js_str(r['player'])},"
        f"team:{js_str(r['team'])},"
        f"y:{js_str(r['yahoo_pos'])},"
        f"nhl:{js_str(r['nhl_pos'])},"
        f"proj:{float(r['proj_fp'])},"
        f"last:{float(last)},"
        f"gp:{int(r.get('last_year_gp') or 0)},"
        f"pgp:{int(r.get('proj_gp') or r.get('exp_gp') or 0)},"
        f"inj:{js_str(r.get('injury') or r.get('flag') or 'healthy')},"
        f"note:{js_str(r.get('note') or '')},"
        f"avoid:{str(bool(r.get('avoid'))).lower()},"
        f"c:{str(bool(r.get('c'))).lower()},"
        f"lw:{str(bool(r.get('lw'))).lower()},"
        f"rw:{str(bool(r.get('rw'))).lower()},"
        f"f:{str(bool(r.get('f'))).lower()},"
        f"d:{str(bool(r.get('d'))).lower()},"
        f"g:{str(bool(r.get('g'))).lower()},"
        f"our:{our_js}"
        "}"
    )

players_js = ",\n  ".join(lines)

tsx = f'''import {{
  Button,
  Callout,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useState,
}} from "cursor/canvas";

type View = "OVR" | "F" | "C" | "LW" | "RW" | "D" | "G";

type Player = {{
  rk: number;
  rf: number | null;
  rc: number | null;
  rl: number | null;
  rr: number | null;
  rd: number | null;
  rg: number | null;
  name: string;
  team: string;
  y: string;
  nhl: string;
  proj: number;
  last: number;
  gp: number;
  pgp: number;
  inj: string;
  note: string;
  avoid: boolean;
  c: boolean;
  lw: boolean;
  rw: boolean;
  f: boolean;
  d: boolean;
  g: boolean;
  our: number | null;
}};

const PLAYERS: Player[] = [
  {players_js}
];

function inView(p: Player, view: View): boolean {{
  if (view === "OVR") return true;
  if (view === "F") return p.f;
  if (view === "C") return p.c;
  if (view === "LW") return p.lw;
  if (view === "RW") return p.rw;
  if (view === "D") return p.d;
  return p.g;
}}

function posNum(p: Player, view: View): number | string {{
  if (view === "OVR") return p.rk;
  if (view === "F") return p.rf ?? "";
  if (view === "C") return p.rc ?? "";
  if (view === "LW") return p.rl ?? "";
  if (view === "RW") return p.rr ?? "";
  if (view === "D") return p.rd ?? "";
  return p.rg ?? "";
}}

function injLabel(inj: string): string {{
  if (inj === "healthy") return "healthy";
  if (inj === "DTD") return "DTD -3 GP";
  if (inj === "IR-LT") return "IR-LT";
  if (inj === "IR-NR") return "IR stash";
  if (inj === "SUSP") return "SUSP -6 GP";
  if (inj === "tiny") return "tiny sample";
  if (inj === "inferred") return "inferred G";
  return inj;
}}

export default function YahooJeffsDraftBoard() {{
  const [view, setView] = useState<View>("OVR");
  const [hideAvoid, setHideAvoid] = useState(true);
  const filtered = PLAYERS.filter((p) => inView(p, view)).filter((p) =>
    hideAvoid ? !p.avoid : true,
  );
  const stash = PLAYERS.filter((p) => p.avoid);
  const nF = PLAYERS.filter((p) => p.f).length;
  const nG = PLAYERS.filter((p) => p.g).length;
  const barkov = PLAYERS.find((p) => p.name === "Aleksander Barkov");

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Jeff's Hockey draft board</H1>
        <Text tone="secondary">
          Mon Sep 28, 2026 · 10-team keeper leftovers · ranked by 2026-27 league
          Fan Pts. Skaters G=7 A=5 +/-=2 PIM=1 PPG=2 PPA=1 SHG=3 GWG=5 SOG=0.5.
          Goalies W=10 L=-7 GA=-2 SV=0.5 SHO=7. Goalies are inferred (no G page
          in the FA paste).
        </Text>
      </Stack>

      <Callout tone="success" title="Barkov is BACK">
        FLA roster, healthy, 82 GP. Yahoo NA was last year's 0 NHL GP, not a
        current skip. 2024-25 mix 20 G / 51 A / 71 pts in 67, scaled to our 80
        NHL pts. Overall #{'{'}barkov?.rk{'}'} · C #{'{'}barkov?.rc{'}'} · {'{'}barkov?.proj.toFixed(1){'}'} FP.
        DTD is 2-5 GP, still draftable — Larkin stays near the top.
      </Callout>

      <Callout tone="neutral" title="Take first among leftover FAs">
        DeBrincat, Larkin (DTD -3 GP), Panarin, Forsberg, Scheifele. Werenski
        if you need D1. Greaves is the first inferred goalie if your 1G slot is
        empty. This scoring pays goals, shots, and goalie saves.
      </Callout>

      <Grid columns={{4}} gap={{16}}>
        <Stat value={{String(PLAYERS.length)}} label="Rows (100 skaters + inferred G)" />
        <Stat value="711.6" label="DeBrincat proj FP" />
        <Stat value={{String(nF)}} label="Forwards on F / Util" />
        <Stat value={{String(nG)}} label="Inferred leftover G" />
      </Grid>

      <Stack gap={{8}}>
        <H2>Position</H2>
        <Row gap={{8}} wrap>
          <Pill active={{view === "OVR"}} onClick={{() => setView("OVR")}}>Overall</Pill>
          <Pill active={{view === "F"}} onClick={{() => setView("F")}}>F</Pill>
          <Pill active={{view === "C"}} onClick={{() => setView("C")}}>C</Pill>
          <Pill active={{view === "LW"}} onClick={{() => setView("LW")}}>LW</Pill>
          <Pill active={{view === "RW"}} onClick={{() => setView("RW")}}>RW</Pill>
          <Pill active={{view === "D"}} onClick={{() => setView("D")}}>D</Pill>
          <Pill active={{view === "G"}} onClick={{() => setView("G")}}>G</Pill>
          <Button
            variant={{hideAvoid ? "primary" : "ghost"}}
            onClick={{() => setHideAvoid((x) => !x)}}
          >
            {{hideAvoid ? "Hide IR/tiny" : "Show IR/tiny"}}
          </Button>
        </Row>
        <Text size="small" tone="tertiary">
          Overall = F+D+G. F = every C/W for Util. LW/RW uses NHL side.
          DTD stays on the board. Hide only IR-NR (Barzal) and tiny-sample
          rookies. Goalie rows are inferred leftovers, not pasted FAs.
        </Text>
      </Stack>

      <Table
        stickyHeader
        striped
        headers={{["Pos#", "Ovr", "Player", "Tm", "Yahoo", "Inj", "Proj GP", "Proj FP", "Last FP", "Note"]}}
        columnAlign={{["right", "right", "left", "left", "left", "left", "right", "right", "right", "left"]}}
        rowTone={{filtered.map((p) =>
          p.inj === "healthy" && p.rk <= 6
            ? "success"
            : p.inj === "DTD" || p.inj === "IR-LT" || p.inj === "SUSP"
              ? "warning"
              : p.g
                ? "info"
                : p.avoid
                  ? "warning"
                  : undefined,
        )}}
        rows={{filtered.map((p) => [
          posNum(p, view),
          p.rk,
          p.name,
          p.team,
          p.y,
          injLabel(p.inj),
          p.pgp,
          p.proj.toFixed(1),
          p.last ? p.last.toFixed(1) : "—",
          p.note,
        ])}}
      />

      <CollapsibleSection title="IR stash / tiny sample" count={{stash.length}}>
        <Text size="small" tone="secondary">
          Barzal is IR-NR (not on NYI active roster). Rookies are tiny-sample,
          not injuries. Marchand IR-LT is on the main board (miss October, still
          draftable mid/late with an IR slot).
        </Text>
        <Table
          headers={{["Ovr", "Player", "Inj", "Proj FP", "Note"]}}
          columnAlign={{["right", "left", "left", "right", "left"]}}
          rows={{stash.map((p) => [
            p.rk,
            p.name,
            injLabel(p.inj),
            p.proj.toFixed(1),
            p.note,
          ])}}
        />
      </CollapsibleSection>

      <Divider />
      <Text size="small" tone="tertiary">
        Live Sep 28: Barkov on FLA roster / healthy. Larkin DTD, DET opener Oct
        2. Rust/Faber/Ovechkin/Duchene DTD, still draftable. McAvoy Yahoo NA is
        a 6-game suspension, not injury. Brady Tkachuk is not in this FA pool
        (kept). Goalie FP converted from Athletic W/L/GA/SV/SHO to this league.
        Assumed kept G: Vasy, Hellebuyck, Shesterkin, Thompson, Oettinger,
        Sorokin, Swayman, Wedgewood. NHL points-only top50 CSV not touched.
      </Text>
    </Stack>
  );
}}
'''

CANVAS.write_text(tsx, encoding="utf-8")
print("wrote", CANVAS, "players", len(rows))
