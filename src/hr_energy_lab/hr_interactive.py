# hr_interactive.py
# Requirements:
#   pip install numpy fitparse matplotlib pyyaml
#
# Usage:
#   - Put this script and profile.yaml in the same folder as your .fit files
#   - Run: python hr_interactive.py
#   - It will list the newest .fit files (count from YAML), ask which to load,
#     copy (minute,HR) CSV to clipboard, and open an interactive plot.

from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import subprocess
import bisect
import statistics as stats

import numpy as np
import fitparse
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import yaml


# ---------------------------------------------------------------------
# FIT reading and file selection
# ---------------------------------------------------------------------

def extract_hr_vs_minutes(fit_path: Path):
    """Return (minutes_from_start, heart_rate_bpm) lists from a .fit file."""
    fitfile = fitparse.FitFile(str(fit_path))

    minutes = []
    hr_vals = []
    t0 = None

    for record in fitfile.get_messages("record"):
        data = {d.name: d.value for d in record}

        ts = data.get("timestamp")
        hr = data.get("heart_rate")

        if ts is None or hr is None:
            continue

        if t0 is None:
            t0 = ts

        dt_min = (ts - t0).total_seconds() / 60.0
        minutes.append(float(dt_min))
        hr_vals.append(float(hr))

    return minutes, hr_vals


def choose_fit_in_folder(
    base_dir: Optional[Path] = None,
    max_files: int = 5,
) -> Path:
    """Show the N newest .fit files; let user choose by index or full filename."""
    if base_dir is None:
        base_dir = Path.cwd()

    fit_files = sorted(
        base_dir.glob("*.fit"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not fit_files:
        raise FileNotFoundError(f"No .fit files found in {base_dir}")

    top_files = fit_files[:max_files]

    print(f"Newest .fit files in {base_dir}:")
    for i, p in enumerate(top_files, 1):
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        print(f"{i}: {p.name}  [{mtime.strftime('%Y-%m-%d %H:%M:%S')}]")

    while True:
        choice = input(
            f"Select by number (1..{len(top_files)}) or type full filename "
            f"(default 1): "
        ).strip()

        if choice == "":
            return top_files[0]

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(top_files):
                return top_files[idx - 1]
            print(f"Number must be between 1 and {len(top_files)}.")
            continue

        candidate = base_dir / choice
        if candidate.exists() and candidate.suffix.lower() == ".fit":
            return candidate

        matches = [p for p in fit_files if p.name == choice]
        if matches:
            return matches[0]

        print("Filename not found. Use exact name, for example crossfit.fit.")


def copy_csv_to_clipboard(minutes, hr_vals):
    """Copy CSV (minute,heart_rate_bpm) to Windows clipboard."""
    lines = ["minute,heart_rate_bpm"]
    lines += [f"{m:.2f},{h:.0f}" for m, h in zip(minutes, hr_vals)]
    csv_text = "\n".join(lines)

    try:
        subprocess.run(
            "clip",
            input=csv_text,
            text=True,
            check=True,
            shell=True,
        )
        print(f"Copied {len(minutes)} rows to clipboard.")
    except Exception as e:
        print(f"Could not copy to clipboard: {e}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_nearest_index(x, xs):
    """Return index of xs closest to x. xs must be sorted."""
    pos = bisect.bisect_left(xs, x)
    if pos == 0:
        return 0
    if pos >= len(xs):
        return len(xs) - 1
    before = pos - 1
    after = pos
    if x - xs[before] <= xs[after] - x:
        return before
    return after


def compute_hr_stats(hr_vals) -> Optional[Tuple[float, float, float, float]]:
    """Return (mean, median, min, max) for HR, or None if empty."""
    if not hr_vals:
        print("No heart rate data for statistics.")
        return None

    mean_hr = stats.fmean(hr_vals)
    median_hr = stats.median(hr_vals)
    min_hr = min(hr_vals)
    max_hr = max(hr_vals)

    print(
        "Heart rate stats [bpm]: "
        f"mean={mean_hr:.1f}, median={median_hr:.1f}, "
        f"min={min_hr:.0f}, max={max_hr:.0f}"
    )

    return mean_hr, median_hr, min_hr, max_hr


# ---------------------------------------------------------------------
# Zones and profile
# ---------------------------------------------------------------------

def build_zones_from_config(
    zones_cfg, hr_rest: Optional[float], hr_max: Optional[float]
) -> List[Dict[str, Any]]:
    """
    Build zones as list of dicts with keys: name, low, high (bpm).
    """
    zones: List[Dict[str, Any]] = []

    if isinstance(zones_cfg, list) and zones_cfg:
        for i, z in enumerate(zones_cfg):
            if not isinstance(z, dict):
                continue

            name = str(z.get("name", f"Z{i+1}"))
            low = None
            high = None

            if "low_bpm" in z or "high_bpm" in z:
                low = z.get("low_bpm", None)
                high = z.get("high_bpm", None)

            elif (
                ("low_pct" in z or "high_pct" in z)
                and hr_rest is not None
                and hr_max is not None
            ):
                hrr = hr_max - hr_rest
                low_pct = z.get("low_pct", None)
                high_pct = z.get("high_pct", None)
                if low_pct is not None:
                    low = hr_rest + float(low_pct) * hrr
                if high_pct is not None:
                    high = hr_rest + float(high_pct) * hrr

            else:
                continue

            zones.append({"name": name, "low": low, "high": high})

    elif hr_rest is not None and hr_max is not None:
        # Fallback simple 5-zone HRR model if no zones in YAML
        hrr = hr_max - hr_rest
        default_ranges = [
            (0.50, 0.60),
            (0.60, 0.70),
            (0.70, 0.80),
            (0.80, 0.90),
            (0.90, 1.00),
        ]
        for i, (lp, hp) in enumerate(default_ranges):
            name = f"Z{i+1}"
            low = hr_rest + lp * hrr
            high = hr_rest + hp * hrr
            zones.append({"name": name, "low": low, "high": high})

    return zones


def load_profile_from_yaml(path: Path) -> Dict[str, Any]:
    """
    Load profile + modeling config from YAML into a dict:

      name, sex, age, weight_kg, height_cm,
      hr_rest, hr_max, zones, hr_error_bpm,
      smoothing_target_durations_min,
      calories_model_frac, calories_include_model_uncertainty,
      recent_fit_files
    """
    if not path.exists():
        raise FileNotFoundError(f"profile YAML not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sex = str(data.get("sex", "M")).upper()
    if sex not in ("M", "F"):
        raise ValueError("profile.yaml: 'sex' must be 'M' or 'F'.")

    try:
        age = int(data["age"])
    except Exception:
        raise ValueError("profile.yaml: 'age' (int) is required.")

    try:
        weight = float(data["weight"])
    except Exception:
        raise ValueError("profile.yaml: 'weight' (float) is required.")

    unit = str(data.get("weight_unit", "kg")).lower()
    if unit.startswith("l"):
        weight_kg = weight * 0.45359237
    else:
        weight_kg = weight

    height_cm = data.get("height_cm", None)
    name = data.get("name", None)

    hr_rest = float(data["hr_rest"]) if "hr_rest" in data else None
    hr_max = float(data["hr_max"]) if "hr_max" in data else None

    zones_cfg = data.get("zones", None)
    zones = build_zones_from_config(zones_cfg, hr_rest, hr_max)

    # Device / quantisation error floor in bpm
    hr_error_bpm = float(data.get("hr_error_bpm", 3.0))

    # Smoothing configuration
    smoothing_cfg = data.get("smoothing", {}) or {}
    target_durations_min = smoothing_cfg.get("target_durations_min", None)
    if isinstance(target_durations_min, list):
        try:
            target_durations_min = [float(v) for v in target_durations_min]
        except Exception:
            print("profile.yaml: invalid smoothing.target_durations_min, using defaults.")
            target_durations_min = None
    else:
        target_durations_min = None

    # Calorie modeling configuration
    calories_cfg = data.get("calories", {}) or {}
    model_frac = float(calories_cfg.get("model_frac", 0.2))  # 1σ fractional model error
    include_model_unc = bool(calories_cfg.get("include_model_uncertainty", True))

    # Script-level UI config
    script_cfg = data.get("script", {}) or {}
    recent_fit_files = int(script_cfg.get("recent_fit_files", 5))

    print(
        "Loaded profile from YAML: "
        f"name={name}, sex={sex}, age={age}, "
        f"weight={weight_kg:.1f} kg, hr_rest={hr_rest}, hr_max={hr_max}, "
        f"hr_error_bpm={hr_error_bpm}, model_frac={model_frac}"
    )

    profile = {
        "name": name,
        "sex": sex,
        "age": age,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "hr_rest": hr_rest,
        "hr_max": hr_max,
        "zones": zones,
        "hr_error_bpm": hr_error_bpm,
        "smoothing_target_durations_min": target_durations_min,
        "calories_model_frac": model_frac,
        "calories_include_model_uncertainty": include_model_unc,
        "recent_fit_files": recent_fit_files,
    }
    return profile


def prompt_user_profile_fallback() -> Dict[str, Any]:
    """Fallback profile if YAML missing or invalid."""
    print(
        "Could not load profile.yaml. Using fallback profile "
        "(M, 30y, 75 kg, hr_error_bpm=3, model_frac=0.2)."
    )
    profile = {
        "name": None,
        "sex": "M",
        "age": 30,
        "weight_kg": 75.0,
        "height_cm": None,
        "hr_rest": None,
        "hr_max": None,
        "zones": [],
        "hr_error_bpm": 3.0,
        "smoothing_target_durations_min": None,
        "calories_model_frac": 0.2,
        "calories_include_model_uncertainty": True,
        "recent_fit_files": 5,
    }
    return profile


def get_profile() -> Dict[str, Any]:
    """Load profile from profile.yaml if possible, otherwise fallback."""
    profile_path = Path.cwd() / "profile.yaml"
    try:
        return load_profile_from_yaml(profile_path)
    except Exception as e:
        print(f"Could not load profile.yaml ({e}).")
        return prompt_user_profile_fallback()


# ---------------------------------------------------------------------
# Calories with strict linear error propagation from 1σ HR + device floor
# ---------------------------------------------------------------------

def estimate_calories_kcal_strict(
    minutes,
    hr_mean,
    hr_sigma_local,
    sex: str,
    age: int,
    weight_kg: float,
    hr_error_bpm: float,
    model_frac: float = 0.2,
    include_model_uncertainty: bool = True,
) -> Tuple[float, float]:
    """Estimate calories and total 1σ error from HR trace.

    hr_error_bpm:
        Device / quantisation 1σ floor for HR readings in bpm.

    model_frac:
        1σ fractional model uncertainty (eg 0.2 for 20 percent),
        read from profile.yaml["calories"]["model_frac"].

    include_model_uncertainty:
        If false, only HR-trace noise is propagated, model_frac ignored.
    """
    t = np.asarray(minutes, float)
    hr_mean = np.asarray(hr_mean, float)
    hr_sigma_local = np.asarray(hr_sigma_local, float)

    if len(t) < 2 or len(hr_mean) < 2:
        return 0.0, 0.0
    if hr_sigma_local.shape != hr_mean.shape:
        raise ValueError("hr_sigma_local must have same length as hr_mean")

    # Midpoint times and HR
    dts_min = np.diff(t)
    hr_mid = 0.5 * (hr_mean[1:] + hr_mean[:-1])
    sigma_mid_local = 0.5 * (hr_sigma_local[1:] + hr_sigma_local[:-1])

    # Effective per-point 1σ at each midpoint: local noise ⊕ device floor
    sigma_mid_eff = np.sqrt(sigma_mid_local**2 + float(hr_error_bpm) ** 2)

    # HR → kcal/min mapping and its derivative wrt HR
    if sex.upper() == "F":
        rate = (
            -20.4022
            + 0.4472 * hr_mid
            - 0.1263 * weight_kg
            + 0.074 * age
        ) / 4.184
        c = 0.4472 / 4.184  # d(rate)/d(HR) in kcal·min⁻¹·bpm⁻¹
    else:
        rate = (
            -55.0969
            + 0.6309 * hr_mid
            + 0.1988 * weight_kg
            + 0.2017 * age
        ) / 4.184
        c = 0.6309 / 4.184

    # Clip negative rates to zero
    rate = np.clip(rate, 0.0, None)

    # Baseline calories from HR trace
    baseline_kcal = float(np.sum(rate * dts_min))

    # Strict propagated HR-trace uncertainty σ_HR
    var_E_hr = float(np.sum((c * dts_min * sigma_mid_eff) ** 2))
    sigma_E_hr = float(np.sqrt(var_E_hr))

    # Model-uncertainty term: fractional error on total calories
    if include_model_uncertainty and model_frac > 0:
        sigma_E_model = model_frac * baseline_kcal
    else:
        sigma_E_model = 0.0

    # Total 1σ uncertainty combining HR noise and model error
    sigma_E_total = float(np.sqrt(sigma_E_hr ** 2 + sigma_E_model ** 2))

    if sigma_E_model > 0:
        print(
            "Calories (HR + model uncertainty): "
            f"E = {baseline_kcal:.0f} ± {sigma_E_total:.0f} kcal "
            f"(σ_HR={sigma_E_hr:.0f}, σ_model={sigma_E_model:.0f}, "
            f"hr_error_bpm={hr_error_bpm:.1f})"
        )
    else:
        print(
            "Calories (HR-trace uncertainty only): "
            f"E = {baseline_kcal:.0f} ± {sigma_E_total:.0f} kcal "
            f"(σ_HR={sigma_E_hr:.0f}, hr_error_bpm={hr_error_bpm:.1f})"
        )
    return baseline_kcal, sigma_E_total


# ---------------------------------------------------------------------
# Smoothing and zones
# ---------------------------------------------------------------------

def compute_smoothed_auto(minutes, hr_vals, target_durations_min=None):
    """
    Choose smoothing window via LOOCV and compute smoothed HR and local std.

    target_durations_min (from YAML):
        Sequence of window durations in minutes to test, eg
        [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0].
    """
    n = len(hr_vals)
    if n == 0:
        return [], [], None, None
    if n < 5:
        return list(hr_vals), [0.0] * n, None, None

    x = np.asarray(hr_vals, float)
    minutes = np.asarray(minutes, float)

    dts = np.diff(minutes)
    pos_dts = dts[dts > 0]

    if target_durations_min is None:
        target_durations_min = np.array(
            [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
        )
    else:
        target_durations_min = np.asarray(target_durations_min, float)

    candidate_ws = set()
    if pos_dts.size:
        med_dt = float(np.median(pos_dts))
        for dur in target_durations_min:
            w = int(round(dur / med_dt))
            if w < 3:
                w = 3
            if w % 2 == 0:
                w += 1
            if w <= n:
                candidate_ws.add(w)

    if not candidate_ws:
        base = min(n if n % 2 == 1 else n - 1, 15)
        if base < 3:
            base = 3
        candidate_ws = {base}

    candidate_ws = sorted(candidate_ws)

    cum = np.cumsum(np.insert(x, 0, 0.0))

    def cv_mse(w: int) -> float:
        half = w // 2
        sqerr = 0.0
        count = 0
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            length = hi - lo
            if length <= 1:
                continue
            window_sum = cum[hi] - cum[lo]
            loo_mean = (window_sum - x[i]) / (length - 1)
            diff = x[i] - loo_mean
            sqerr += diff * diff
            count += 1
        return sqerr / max(count, 1)

    best_w = min(candidate_ws, key=cv_mse)

    half = best_w // 2
    smoothed = []
    local_std = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = x[lo:hi]
        mu = float(win.mean())
        smoothed.append(mu)
        if win.size > 1:
            sigma = float(win.std(ddof=0))
        else:
            sigma = 0.0
        local_std.append(sigma)

    if pos_dts.size:
        med_dt = float(np.median(pos_dts))
        eff_dur_min = best_w * med_dt
    else:
        eff_dur_min = None

    return smoothed, local_std, best_w, eff_dur_min


def compute_zone_stats(
    minutes, hr_vals, zones: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Compute time in each HR zone based on smoothed HR.
    """
    if not zones:
        print("No zones defined in profile; zone labels will be skipped.")
        return [], 0.0
    if len(minutes) < 2 or len(hr_vals) < 2:
        return [], 0.0

    t = np.asarray(minutes, float)
    hr = np.asarray(hr_vals, float)

    dts = np.diff(t)
    hr_mid = 0.5 * (hr[1:] + hr[:-1])
    total_time = float(np.sum(dts))

    zone_stats: List[Dict[str, Any]] = []

    for z in zones:
        name = z.get("name", "Z")
        low = z.get("low", None)
        high = z.get("high", None)

        mask = np.ones_like(hr_mid, dtype=bool)
        if low is not None:
            mask &= hr_mid >= low
        if high is not None:
            mask &= hr_mid < high

        time_min = float(np.sum(dts[mask])) if total_time > 0 else 0.0
        pct = (time_min / total_time * 100.0) if total_time > 0 else 0.0

        zone_stats.append(
            {"name": name, "time_min": time_min, "pct": pct, "low": low, "high": high}
        )

    return zone_stats, total_time


# ---------------------------------------------------------------------
# Plotting + interaction
# ---------------------------------------------------------------------

def plot_interactive(
    minutes,
    hr_vals,
    smoothed_hr,
    local_std,
    hr_stats,
    kcal_estimate,
    kcal_sigma,
    zones,
    zone_stats,
):
    if not minutes:
        print("No valid HR records found.")
        return

    minutes_arr = np.asarray(minutes, float)
    smoothed = np.asarray(smoothed_hr, float)
    sigma = np.asarray(local_std, float)

    lower = smoothed - sigma
    upper = smoothed + sigma

    y_full_min = float(lower.min())
    y_full_max = float(upper.max())
    x_full_min = float(minutes_arr.min())
    x_full_max = float(minutes_arr.max())
    y_margin = 0.05 * max(1.0, y_full_max - y_full_min)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.subplots_adjust(right=0.8)

    # Zone shading
    if zones:
        zone_colors = [
            "#e0f3db",
            "#a8ddb5",
            "#7bccc4",
            "#4eb3d3",
            "#2b8cbe",
            "#fdae6b",
            "#fb6a4a",
        ]
        for i, z in enumerate(zones):
            low = z.get("low", None)
            high = z.get("high", None)

            low_plot = y_full_min if low is None else low
            high_plot = y_full_max if high is None else high

            ax.axhspan(
                low_plot,
                high_plot,
                facecolor=zone_colors[i % len(zone_colors)],
                alpha=0.25,
                zorder=0,
                label="_nolegend_",
            )

    # Data
    ax.plot(minutes, hr_vals, alpha=0.3, linewidth=1.0, label="Raw HR", zorder=2)
    ax.plot(minutes, smoothed_hr, linewidth=1.5, label="Smoothed HR", zorder=3)
    ax.fill_between(minutes, lower, upper, alpha=0.2, label="±1σ (local)", zorder=1)

    ax.set_xlabel("Minutes from start")
    ax.set_ylabel("Heart rate [bpm]")
    ax.set_title("Heart rate vs time")

    ax.set_xlim(x_full_min, x_full_max)
    ax.set_ylim(y_full_min - y_margin, y_full_max + y_margin)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best")

    # Stats box (fixed in axes coords)
    y_axis_min, y_axis_max = ax.get_ylim()
    y_span = max(y_axis_max - y_axis_min, 1e-6)

    stats_lines = []
    if hr_stats is not None:
        mean_hr, median_hr, min_hr, max_hr = hr_stats
        stats_lines.append(f"Mean:   {mean_hr:.1f} bpm")
        stats_lines.append(f"Median: {median_hr:.1f} bpm")
        stats_lines.append(f"Min:    {min_hr:.0f} bpm")
        stats_lines.append(f"Max:    {max_hr:.0f} bpm")

    if kcal_estimate is not None:
        if kcal_sigma is not None and kcal_sigma > 0:
            stats_lines.append(
                f"Est. cals: {kcal_estimate:.0f} ± {kcal_sigma:.0f} kcal"
            )
        else:
            stats_lines.append(f"Est. cals: {kcal_estimate:.0f} kcal")

    if stats_lines:
        stats_text = "\n".join(stats_lines)
        ax.text(
            0.01,
            0.99,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            zorder=4,
        )

    # Dynamic zone labels on the right side (outside), updated on zoom
    zone_line_artists: List[Any] = []
    zone_text_artists: List[Any] = []

    def update_zone_labels():
        nonlocal zone_line_artists, zone_text_artists

        # Remove existing artists
        for art in zone_line_artists + zone_text_artists:
            try:
                art.remove()
            except ValueError:
                pass
        zone_line_artists = []
        zone_text_artists = []

        if not zone_stats:
            fig.canvas.draw_idle()
            return

        y_min, y_max = ax.get_ylim()
        span = max(y_max - y_min, 1e-6)

        for zstat in zone_stats:
            low = zstat.get("low", None)
            high = zstat.get("high", None)

            low_plot = y_min if low is None else low
            high_plot = y_max if high is None else high

            # Skip if zone is completely outside current visible range
            if high_plot < y_min or low_plot > y_max:
                continue

            low_clip = max(low_plot, y_min)
            high_clip = min(high_plot, y_max)
            y_mid = 0.5 * (low_clip + high_clip)

            y_frac = (y_mid - y_min) / span
            y_frac = min(max(y_frac, 0.0), 1.0)

            label = f"{zstat['name']}: {zstat['time_min']:.1f} min ({zstat['pct']:.0f}%)"

            line = ax.plot(
                [1.0, 1.02],
                [y_frac, y_frac],
                transform=ax.transAxes,
                linestyle="--",
                linewidth=0.8,
                color="k",
                zorder=5,
                clip_on=False,
            )[0]

            text = ax.text(
                1.03,
                y_frac,
                label,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=8,
                zorder=5,
                clip_on=False,
            )

            zone_line_artists.append(line)
            zone_text_artists.append(text)

        fig.canvas.draw_idle()

    # Initial draw of zone labels
    update_zone_labels()

    # Hover line + annotation
    vline = ax.axvline(minutes_arr[0], linestyle="--", zorder=4)
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annotation(idx):
        x = minutes_arr[idx]
        y = smoothed[idx]
        s = sigma[idx]
        vline.set_xdata([x, x])
        annot.xy = (x, y)
        annot.set_text(f"{x:.2f} min\n{y:.0f} ± {s:.1f} bpm")
        annot.set_visible(True)

    def on_move(event):
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        x = event.xdata
        if x is None:
            return

        idx = find_nearest_index(x, minutes_arr)
        update_annotation(idx)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # Left drag zoom (rectangle)
    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        ax.set_xlim(min(x1, x2), max(x1, x2))
        ax.set_ylim(min(y1, y2), max(y1, y2))
        update_zone_labels()

    RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=0.01,
        minspany=0.01,
        spancoords="data",
        interactive=False,
    )

    # Double-click in, right-click out (factor 2)
    def on_click(event):
        if event.inaxes is not ax:
            return

        if event.button == 1 and getattr(event, "dblclick", False):
            # Zoom in by factor 2 around click
            x_center = event.xdata if event.xdata is not None else 0.5 * (x_full_min + x_full_max)
            y_center = event.ydata if event.ydata is not None else 0.5 * (y_full_min + y_full_max)
            cur_xmin, cur_xmax = ax.get_xlim()
            cur_ymin, cur_ymax = ax.get_ylim()
            span_x = (cur_xmax - cur_xmin) * 0.5
            span_y = (cur_ymax - cur_ymin) * 0.5

            new_xmin = max(x_full_min, x_center - span_x / 2)
            new_xmax = min(x_full_max, x_center + span_x / 2)
            new_ymin = max(y_full_min - y_margin, y_center - span_y / 2)
            new_ymax = min(y_full_max + y_margin, y_center + span_y / 2)

            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)
            update_zone_labels()

        elif event.button == 3:
            # Zoom out by factor 2 around click
            cur_xmin, cur_xmax = ax.get_xlim()
            cur_ymin, cur_ymax = ax.get_ylim()
            x_center = event.xdata if event.xdata is not None else 0.5 * (cur_xmin + cur_xmax)
            y_center = event.ydata if event.ydata is not None else 0.5 * (cur_ymin + cur_ymax)

            span_x = (cur_xmax - cur_xmin) * 2.0
            span_y = (cur_ymax - cur_ymin) * 2.0

            new_xmin = max(x_full_min, x_center - span_x / 2)
            new_xmax = min(x_full_max, x_center + span_x / 2)
            new_ymin = max(y_full_min - y_margin, y_center - span_y / 2)
            new_ymax = min(y_full_max + y_margin, y_center + span_y / 2)

            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)
            update_zone_labels()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Also update zone labels if toolbar zoom/pan changes limits
    ax.callbacks.connect("ylim_changed", lambda ax: update_zone_labels())
    ax.callbacks.connect("xlim_changed", lambda ax: update_zone_labels())

    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    profile = get_profile()

    fit_path = choose_fit_in_folder(
        Path.cwd(),
        max_files=profile.get("recent_fit_files", 5),
    )
    print(f"Using FIT file: {fit_path}")

    minutes, hr_vals = extract_hr_vs_minutes(fit_path)

    smoothed_hr, local_std, best_w, eff_dur_min = compute_smoothed_auto(
        minutes,
        hr_vals,
        target_durations_min=profile.get("smoothing_target_durations_min"),
    )
    if best_w is not None:
        if eff_dur_min is not None:
            print(
                f"Smoothing window: {best_w} samples "
                f"(~{eff_dur_min * 60:.1f} s) selected by LOOCV."
            )
        else:
            print(f"Smoothing window: {best_w} samples selected by LOOCV.")

    hr_stats = compute_hr_stats(hr_vals)

    kcal_est, kcal_sigma = estimate_calories_kcal_strict(
        minutes,
        smoothed_hr,
        local_std,
        profile["sex"],
        profile["age"],
        profile["weight_kg"],
        profile["hr_error_bpm"],
        model_frac=profile.get("calories_model_frac", 0.2),
        include_model_uncertainty=profile.get(
            "calories_include_model_uncertainty", True
        ),
    )

    zones = profile.get("zones", [])
    zone_stats, _ = compute_zone_stats(minutes, smoothed_hr, zones)

    copy_csv_to_clipboard(minutes, hr_vals)

    plot_interactive(
        minutes,
        hr_vals,
        smoothed_hr,
        local_std,
        hr_stats,
        kcal_est,
        kcal_sigma,
        zones,
        zone_stats,
    )


if __name__ == "__main__":
    main()
