#!/usr/bin/env python3
import json, os

def main() -> None:
    p = os.path.join(os.path.dirname(__file__), "discoveries_manifest.json")
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    print("=" * 88)
    print(f"  PROJECT ARJUNA ({d['architecture']}) — INTERACTIVE DISCOVERY CONSOLE")
    print("=" * 88)
    for k, v in d["mastered_universes"].items():
        print(f"  [Mastered] {k:42s} -> Acc: {v['accuracy']} (CE: {v['ce_loss']})")
    c2 = d["bsd_elliptic_curve_proofs"]["curve_2_million_sweep_champion"]
    print(f"  [BSD Curve] {c2['equation']} | Reg = +{c2['fundamental_regulator']} | {c2['integral_points_spanned']}")
    s = d["erdos_turan_singer_sidon_32_in_1000"]
    print(f"  [Sidon Set] |S| = {s['size']} in [1..1000] ({s['unique_pairwise_sums']} unique sums)")
    print("=" * 88)

if __name__ == "__main__":
    main()
