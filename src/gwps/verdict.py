"""Floors from controls, clusters on the mesh, and the BAKE / BUILD / INCONCLUSIVE rules."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MARGIN_DEG = 1.0
INCONCLUSIVE_FLOOR_DEG = 3.0
MRL_BUILD = 0.7
MIN_VIEWS = 2
MIN_BASELINE_DEG = 50.0   # azimuth span of a face's views needed to judge consistency
VERTICAL_FLAG = 0.8       # vertical share of a cluster's disagreement that flags the blind spot
ROW_COVERAGE_MIN = 0.95   # share of the garment's area whose camera elevation board placements must span


def vertical_fraction(dmean):
    """Share of each face's mean disagreement vector (world frame) that lies along the turntable
    axis, d_z^2 / |d|^2; NaN where it is zero. Turntable steps never move a face in image row,
    so a light-fixed bias that tilts normals up or down with the row looks garment-fixed to the
    consistency test (the known blind spot); a BUILD resting on such disagreement is suspect."""
    d = np.asarray(dmean, np.float64)
    n2 = (d ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n2 > 0, d[..., 2] ** 2 / n2, np.nan)


def corrected_mrl(R, n):
    """Mean resultant length corrected for the number of samples: sqrt((n R^2 - 1) / (n - 1)).
    E[R^2] = 1/n for directions that are random across samples, so this is ~0 at chance
    (the raw R is 0.67 / 0.54 / 0.47 / 0.42 at chance for n = 2..5) and 1 for a fixed one.
    The samples are 30 deg turntable-angle bins, not views (see compare.consistency_bins):
    nearby views are not independent, and counting them shrank the correction enough for a
    light-fixed bias to pass as BUILD with 10 deg steps."""
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = (n * np.asarray(R, float) ** 2 - 1) / (n - 1)
    return np.where(n >= 2, np.sqrt(np.clip(r2, 0, None)), np.nan)
VERDICT_MIN_SCALE_MM = 5


def floor_from_thetas(thetas_by_scale, q=95):
    """{s: 1-D array of theta_s over valid control pixels} -> {s: q-th percentile}."""
    return {s: float(np.nanpercentile(v, q)) if np.isfinite(v).any() else float("nan")
            for s, v in thetas_by_scale.items()}


def _spans(placements):
    return [(p["name"], p["elev"]) for p in placements or [] if p.get("elev")]


def _covered(e, spans):
    inside = np.zeros(np.shape(e), bool)
    for _, (lo, hi) in spans:
        inside |= (e >= lo) & (e <= hi)
    return inside


def elev_deg(e):
    """Elevation Y/Z (positive down) as degrees below the optical axis."""
    return float(np.degrees(np.arctan(e)))


def row_coverage(mesh, faces_res, garment, placements, min_views=MIN_VIEWS, min_gap_area=0.005):
    """How much of the garment the board placements cover in camera elevation (the blind spot).

    A light-fixed bias that tilts normals up or down with the image row passes the consistency
    test, because turntable steps never move a face in row. The fabric floor can only contain such
    a bias where a board placement sat at that height in the image, so a garment row outside every
    placement's span (stage 6, <kind>_placements) is a row whose floor was never measured.
    Elevation (Y/Z, compare.compare_view) rather than the pixel row, so stage 1's crop and the
    controls' own undistorted camera do not matter. Returns None without placement spans or
    per-face elevations (older stage 5 / 6 output).

    -> {"covered_area_fraction", "pass" (>= ROW_COVERAGE_MIN), "garment_elev" (1st-99th
    percentile), "uncovered_elev": [[lo, hi], ...] (gaps holding >= min_gap_area of the garment),
    "placements": [{"name", "elev"}]}."""
    spans = _spans(placements)
    if not spans or "elev" not in faces_res:
        return None
    elig = garment & (faces_res["n_views_any"] >= min_views) & np.isfinite(faces_res["elev"])
    if not elig.any():
        return None
    e, A = faces_res["elev"][elig].astype(np.float64), mesh.area_faces[elig]
    inside = _covered(e, spans)
    frac = float(A[inside].sum() / A.sum())
    # gaps: the complement of the spans' union within the garment's elevation range
    edges = sorted(spans, key=lambda x: x[1][0])
    lo_g, hi_g = float(e.min()), float(e.max())
    gaps, cur = [], lo_g
    for _, (lo, hi) in edges:
        if lo > cur:
            gaps.append([cur, min(lo, hi_g)])
        cur = max(cur, hi)
    if cur < hi_g:
        gaps.append([cur, hi_g])
    gaps = [g for g in gaps if g[1] > g[0] and A[(e >= g[0]) & (e <= g[1])].sum() >= min_gap_area * A.sum()]
    return {"covered_area_fraction": frac, "pass": frac >= ROW_COVERAGE_MIN,
            "garment_elev": [float(x) for x in np.percentile(e, [1, 99])], "uncovered_elev": gaps,
            "placements": [{"name": n, "elev": list(sp)} for n, sp in spans]}


def cluster_elevation(c, faces_res, mesh, placements, s, margin=MARGIN_DEG):
    """Annotate a cluster with where it sits in the camera's elevation, how much of it the board
    placements cover, and the highest floor (at its scale) among the placements that overlap it:
    the disagreement a light-fixed bias produced on the board at that height."""
    spans = _spans(placements)
    if not spans or "elev" not in faces_res:
        return c
    f = c["faces"]
    e = faces_res["elev"][f].astype(np.float64)
    ok = np.isfinite(e)
    if not ok.any():
        return c
    lo, hi = (float(x) for x in np.percentile(e[ok], [5, 95]))
    A = mesh.area_faces[f][ok]
    over = [p for p in placements if p.get("elev") and p["elev"][0] <= hi and p["elev"][1] >= lo
            and str(s) in p.get("floor_deg", {})]
    fh = max((p["floor_deg"][str(s)] for p in over), default=None)
    c.update({"elev": [lo, hi], "elev_covered_fraction": float(A[_covered(e[ok], spans)].sum() / A.sum()),
              "height_placements": [p["name"] for p in over], "floor_at_height_deg": fh,
              "exceeds_floor_at_height": None if fh is None else bool(c["theta_mean_deg"] > fh + margin)})
    return c


def find_clusters(mesh, faces_res, garment, floor, s, margin=MARGIN_DEG, min_views=MIN_VIEWS):
    """Connected sets of eligible faces with theta_s > floor_s + margin and area >= (2s)^2."""
    th = faces_res[f"theta_{s}"]
    elig = garment & (faces_res[f"n_views_{s}"] >= min_views) & np.isfinite(th)
    hot = elig & (th > floor + margin)
    if not hot.any():
        return []
    adj = mesh.face_adjacency
    keep = hot[adj[:, 0]] & hot[adj[:, 1]]
    a = adj[keep]
    F = len(mesh.faces)
    g = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(F, F))
    _, labels = connected_components(g, directed=False)
    area_mm2 = mesh.area_faces * 1e6
    clusters = []
    for lab in np.unique(labels[hot]):
        f = np.flatnonzero(hot & (labels == lab))
        A = area_mm2[f].sum()
        if A < (2 * s) ** 2:
            continue
        wA = area_mm2[f]
        nv = faces_res[f"n_views_{s}"][f]
        nb = faces_res[f"n_bins_{s}"][f] if f"n_bins_{s}" in faces_res else nv   # older stage-5 output
        R_raw = faces_res[f"mrl_{s}"][f]
        cons = (faces_res["azimuth_span_deg"][f] >= MIN_BASELINE_DEG) & (nb >= 2)
        mrl = (float(np.average(corrected_mrl(R_raw[cons], nb[cons]), weights=wA[cons]))
               if cons.any() else None)
        vert = None
        if f"dmean_{s}" in faces_res:                                       # older stage-5 output has none
            v = vertical_fraction(faces_res[f"dmean_{s}"][f])
            ok = np.isfinite(v)
            vert = float(np.average(v[ok], weights=wA[ok])) if ok.any() else None
        clusters.append({
            "vertical_fraction": vert,
            "scale_mm": s, "faces": f, "area_mm2": float(A),
            "theta_mean_deg": float(np.average(th[f], weights=wA)),
            "theta_max_deg": float(th[f].max()),
            "mrl": mrl,                                   # None = no face with enough baseline
            "mrl_raw": float(np.average(R_raw, weights=wA)),
            "views_median": float(np.median(nv)), "bins_median": float(np.median(nb)),
            "baseline_area_fraction": float(wA[cons].sum() / A),
            "centroid_world_m": mesh.triangles_center[f].mean(0).tolist(),
        })
    return clusters


def concavity_fraction(mesh, faces, s_mm, _cache={}, max_points=200_000):
    """Area fraction of the faces that are concave at scale s (neighbours within 2s lie on
    the outward-normal side of the face). Interreflections live in concavities.
    The neighbourhood's mean is taken over at most max_points face centres drawn in proportion to
    area, so a GW mesh costs what a small one does: on a 10M-face garment a 0.1 m ball (s = 50 mm)
    holds ~500k faces, and 2000 such queries over every face would be ~10^9 neighbours."""
    key = id(mesh)
    if key not in _cache:
        _cache.clear()
        F = len(mesh.faces)
        if F > max_points:
            a = mesh.area_faces
            idx = np.random.default_rng(0).choice(F, max_points, replace=True, p=a / a.sum())
        else:
            idx = np.arange(F)
        _cache[key] = cKDTree(mesh.triangles_center[idx])
    tree = _cache[key]
    c = mesh.triangles_center
    n = mesh.face_normals
    r = 2 * s_mm / 1000.0
    sub = faces if len(faces) <= 2000 else np.random.default_rng(0).choice(faces, 2000, replace=False)
    conc = []
    for f in sub:
        nb = tree.query_ball_point(c[f], r)
        if nb:
            conc.append(np.dot(n[f], tree.data[nb].mean(0) - c[f]) > 0)
    return float(np.mean(conc)) if conc else 0.0


def decide(mesh, faces_res, garment, fabric_floor, scales, sphere_floor=None, placements=None):
    """Apply the rules in order. Returns a dict with verdict, reasons and clusters.
    placements: stage 6's fabric_placements (elevation spans and floors per board placement),
    for the row-coverage check of the blind spot; it reports, it does not change the rules."""
    big = [s for s in scales if s >= VERDICT_MIN_SCALE_MM]
    bin_deg = float(faces_res["consistency_bin_deg"]) if "consistency_bin_deg" in faces_res else 0.0
    out = {"fabric_floor_deg": fabric_floor, "sphere_floor_deg": sphere_floor, "clusters": {},
           "consistency_bin_deg": bin_deg, "row_coverage": row_coverage(mesh, faces_res, garment, placements)}
    for s in scales:
        cl = find_clusters(mesh, faces_res, garment, fabric_floor[s], s)
        for c in cl:
            c["concave_fraction"] = concavity_fraction(mesh, c["faces"], s)
            c["silhouette_fraction"] = float(np.average(
                faces_res["mean_view_angle_deg"][c["faces"]] > 60.0, weights=mesh.area_faces[c["faces"]]))
            cluster_elevation(c, faces_res, mesh, placements, s)
        out["clusters"][s] = cl
    if any(fabric_floor[s] > INCONCLUSIVE_FLOOR_DEG for s in big):
        out["verdict"] = "INCONCLUSIVE"
        out["reason"] = f"fabric floor above {INCONCLUSIVE_FLOOR_DEG} deg at some s >= 5 mm"
    elif not any(out["clusters"][s] for s in big):
        out["verdict"] = "BAKE"
        out["reason"] = "no cluster at any s >= 5 mm"
    elif any(c["mrl"] is not None and c["mrl"] >= MRL_BUILD for s in big for c in out["clusters"][s]):
        out["verdict"] = "BUILD candidate"
        out["reason"] = f"a cluster at s >= 5 mm has mean resultant length >= {MRL_BUILD}"
        build = [c for s in big for c in out["clusters"][s] if c["mrl"] is not None and c["mrl"] >= MRL_BUILD]
        vert = [c for c in build if c["vertical_fraction"] is not None and c["vertical_fraction"] >= VERTICAL_FLAG]
        if vert:
            w = {"clusters": len(vert), "of": len(build), "all": len(vert) == len(build)}
            if out["row_coverage"] is not None:
                # uncovered: most of the cluster lies at camera elevations no board placement spanned,
                # so nothing measured the row-dependent bias there. within_height_floor: covered, and
                # the board at that height disagreed as much (theta <= its floor + margin).
                w["uncovered"] = sum(c.get("elev_covered_fraction", 0.0) < 0.5 for c in vert)
                w["within_height_floor"] = sum(c.get("elev_covered_fraction", 0.0) >= 0.5
                                               and c.get("exceeds_floor_at_height") is False for c in vert)
            out["vertical_warning"] = w
    else:
        out["verdict"] = "UNEXPLAINED"
        undet = all(c["mrl"] is None for s in big for c in out["clusters"][s])
        out["reason"] = ("clusters exist but no face in them was seen over a wide enough baseline to judge consistency"
                         if undet else "clusters exist but rotate with the lights (mean resultant length < 0.7)")
    return out


def _fmt_height(c):
    hm, hf = c.get("height_mm", False), c.get("height_fine_mm")
    if hm is False:
        out = ""
    elif hm is None:
        out = ", implied height n/a (no view within 45 deg of face-on with a ring around it)"
    else:
        out = (", implied height " + (f"{hf['peak_mm']:+.2f} mm at 2 mm, " if hf else "")
               + f"{hm['peak_mm']:+.2f} mm at s (cluster median {hm['median_mm']:+.2f}; {hm['views']} views)")
    if "elev" not in c:
        return out
    fh = c["floor_at_height_deg"]
    return out + (f", elevation {elev_deg(c['elev'][0]):.1f} to {elev_deg(c['elev'][1]):.1f} deg "
                  f"(board-covered {c['elev_covered_fraction']:.2f}; floor at this height "
                  + ("n/a" if fh is None else f"{fh:.2f} deg from {', '.join(c['height_placements'])}") + ")")


def _fmt_mrl(cl):
    v = [c["mrl"] for c in cl if c["mrl"] is not None]
    return f"{max(v):.2f}" if v else "n/a"


def verdict_markdown(res, scales):
    L = [f"# Verdict: **{res['verdict']}**", "", res["reason"], ""]
    w = res.get("vertical_warning")
    rc = res.get("row_coverage")
    if w:
        L += [f"**Warning: {'all' if w['all'] else w['clusters']} of the {w['of']} cluster(s) behind this BUILD "
              f"{'are' if w['all'] or w['clusters'] > 1 else 'is'} mostly vertical** (vertical share >= {VERTICAL_FLAG}). "
              "Turntable steps never move a face in image row, so a light-fixed bias that tilts normals up or "
              "down with the row passes the consistency test (the known blind spot). Prefer evidence from "
              "clusters that are not vertical" + (" (none here)." if w["all"] else ".")]
        if rc is None:
            L += ["Board-placement heights are unknown (floors.json has no fabric_placements: rerun stage 6), "
                  "so whether the board ever sat at these clusters' rows is unchecked."]
        else:
            if w.get("uncovered"):
                L += [f"**{w['uncovered']} of these vertical clusters lie at rows no board placement covered**: "
                      "nothing measured a row-dependent bias there. Shoot the fabric board at those heights "
                      "(see 'Board coverage' below), rerun stage 6, then this verdict."]
            if w.get("within_height_floor"):
                L += [f"{w['within_height_floor']} of them are covered and do not exceed the floor of the board "
                      f"at their own height + {MARGIN_DEG:.0f} deg: the board there disagreed as much, which "
                      "points to a row-dependent PS bias rather than shape."]
        L += [""]
    if rc is not None:
        gap = "; ".join(f"{elev_deg(a):.1f} to {elev_deg(b):.1f} deg" for a, b in rc["uncovered_elev"]) or "none"
        L += [f"Board coverage (the blind spot): the fabric board's placements span "
              f"{100 * rc['covered_area_fraction']:.1f} % of the garment's area by elevation in the camera "
              f"({'ok' if rc['pass'] else f'**below {100 * ROW_COVERAGE_MIN:.0f} %**'}; garment "
              f"{elev_deg(rc['garment_elev'][0]):.1f} to {elev_deg(rc['garment_elev'][1]):.1f} deg below the "
              f"optical axis, negative = above). Uncovered: {gap}.", ""]
    L += ["| s (mm) | fabric floor (deg) | clusters | largest area (mm^2) | max cluster MRL (corrected) | mean theta in clusters (deg) |",
          "|---|---|---|---|---|---|"]
    for s in scales:
        cl = res["clusters"][s]
        if cl:
            big = max(cl, key=lambda c: c["area_mm2"])
            L.append(f"| {s} | {res['fabric_floor_deg'][s]:.2f} | {len(cl)} | {big['area_mm2']:.0f} | "
                     f"{_fmt_mrl(cl)} | {np.mean([c['theta_mean_deg'] for c in cl]):.2f} |")
        else:
            L.append(f"| {s} | {res['fabric_floor_deg'][s]:.2f} | 0 | - | - | - |")
    L += ["", "Clusters at s >= 5 mm (concave/silhouette = area fractions):", ""]
    for s in scales:
        if s < VERDICT_MIN_SCALE_MM:
            continue
        for c in res["clusters"][s]:
            L.append(f"- s = {s} mm: area {c['area_mm2']:.0f} mm^2, theta {c['theta_mean_deg']:.2f} deg "
                     f"(max {c['theta_max_deg']:.2f}), MRL {'n/a' if c['mrl'] is None else round(c['mrl'], 2)} "
                     f"(raw {c['mrl_raw']:.2f}, views/bins median {c.get('views_median', float('nan')):.0f}/"
                     f"{c.get('bins_median', float('nan')):.0f}, baseline area {c['baseline_area_fraction']:.2f}), "
                     f"vertical {'n/a' if c.get('vertical_fraction') is None else round(c['vertical_fraction'], 2)}, "
                     f"concave {c['concave_fraction']:.2f}, silhouette {c['silhouette_fraction']:.2f}, "
                     f"centroid {np.round(c['centroid_world_m'], 3).tolist()} m" + _fmt_height(c))
    b = res.get("consistency_bin_deg", 0.0)
    how = (f"each face's per-view disagreement directions are averaged within {b:.0f} deg turntable-angle bins, "
           f"and the mean resultant length over bins is corrected for the number of bins"
           if b and b > 0 else "the mean resultant length over views is corrected for the number of views")
    L += ["", f"MRL: {how}. Only faces whose views span >= {MIN_BASELINE_DEG:.0f} deg of azimuth (and >= 2 samples) "
          f"count; 'raw' is the uncorrected value. 'vertical': the area-weighted share of the mean disagreement "
          f"along the turntable axis (d_z^2 / |d|^2); >= {VERTICAL_FLAG} flags the blind spot. 'implied "
          f"height' (stage 7 given stage 5's inputs): the low-passed normal difference integrated over the "
          f"cluster against a ring around it, in views within 45 deg of face-on, median over them; > 0 = the "
          f"surface PS sees lies outside the mesh there (GW low), < 0 = the mesh bulges out. At s, features "
          f"near s read smaller than they are; the 2 mm value comes closer to the full depth.",
          "", "Before acting on a BUILD candidate, rerun GW at `-r 1`."]
    return "\n".join(L) + "\n"
