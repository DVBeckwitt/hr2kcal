# hr_interactive.py
# Requirements:
#   pip install numpy fitparse matplotlib pyyaml
#
# Usage:
#   - Put this script and profile.yaml in the folder you want to use for .fit
#     files (or set that folder once in the GUI).
#   - Run: python hr_interactive.py
#   - It will load your default .fit folder (or prompt once to set one if empty),
#     list the newest .fit files (count from YAML), ask which to load, copy
#     (minute,HR) CSV to clipboard, and open an interactive plot.

from pathlib import Path
import copy
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import subprocess
import bisect
import statistics as stats
import tkinter as tk
from tkinter import filedialog
import sys

import numpy as np
import fitparse
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, TextBox, CheckButtons
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


def prompt_fit_directory(default_dir: Optional[Path] = None) -> Path:
    """Ask the user which folder to scan for .fit files."""
    if default_dir is None:
        default_dir = Path.cwd()

    while True:
        user_input = input(
            f"Folder containing .fit files (leave blank for {default_dir}): "
        ).strip()

        if user_input == "":
            candidate = default_dir
        else:
            candidate = Path(user_input).expanduser()
            if not candidate.is_absolute():
                candidate = (default_dir / candidate).resolve()

        if candidate.exists() and candidate.is_dir():
            return candidate

        print("Directory does not exist. Please enter a valid folder path.")


def select_directory_with_dialog(initial_dir: Path) -> Optional[Path]:
    """Open a native folder chooser; return None if cancelled or unavailable."""

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=str(initial_dir))
        root.destroy()
        if selected:
            return Path(selected).expanduser()
    except Exception:
        return None
    return None


def has_fit_files(path: Path) -> bool:
    """Return True if the directory contains at least one .fit file."""

    return any(path.glob("*.fit"))


def ensure_fit_directory(
    profile: Dict[str, Any],
    profile_yaml_data: Dict[str, Any],
    profile_path: Path,
    fallback_dir: Path,
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """
    Return a valid FIT directory, prompting with a file manager if needed.

    If the resolved default folder has no .fit files, the user is asked to pick
    a replacement before the GUI opens. The chosen folder becomes the in-memory
    default and is written back to profile.yaml.
    """

    current_default = profile.get("default_fit_dir")
    fit_dir = Path(current_default).expanduser() if current_default else fallback_dir

    if not fit_dir.exists():
        fit_dir = fallback_dir

    if has_fit_files(fit_dir):
        return fit_dir, profile, profile_yaml_data

    print(
        f"No .fit files found in {fit_dir}. "
        "Choose a folder to set as the default FIT directory."
    )

    chosen = select_directory_with_dialog(fit_dir)
    if chosen is None:
        chosen = prompt_fit_directory(fit_dir)

    if not has_fit_files(chosen):
        raise FileNotFoundError(
            f"Selected directory {chosen} does not contain any .fit files."
        )

    profile["default_fit_dir"] = str(chosen)
    saved_yaml = save_profile_to_yaml(profile, profile_path, profile_yaml_data)
    return chosen, profile, saved_yaml


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


def load_profile_from_yaml(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    default_fit_dir_raw = script_cfg.get("default_fit_dir")
    default_fit_dir = None
    if isinstance(default_fit_dir_raw, str) and default_fit_dir_raw.strip():
        default_fit_dir = str(Path(default_fit_dir_raw).expanduser())

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
        "zones_cfg_raw": zones_cfg,
        "hr_error_bpm": hr_error_bpm,
        "smoothing_target_durations_min": target_durations_min,
        "calories_model_frac": model_frac,
        "calories_include_model_uncertainty": include_model_unc,
        "recent_fit_files": recent_fit_files,
        "default_fit_dir": default_fit_dir,
    }
    return profile, data


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
        "zones_cfg_raw": [],
        "hr_error_bpm": 3.0,
        "smoothing_target_durations_min": None,
        "calories_model_frac": 0.2,
        "calories_include_model_uncertainty": True,
        "recent_fit_files": 5,
        "default_fit_dir": None,
    }
    return profile


def get_profile(profile_path: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load profile + raw YAML from the given path; otherwise use fallback."""

    resolved_path = profile_path if profile_path is not None else Path.cwd() / "profile.yaml"
    try:
        profile, yaml_data = load_profile_from_yaml(resolved_path)
        return profile, yaml_data
    except Exception as e:
        print(f"Could not load profile.yaml ({e}).")
        return prompt_user_profile_fallback(), {}


def build_profile_yaml(profile_state: Dict[str, Any], base_yaml: Dict[str, Any]) -> Dict[str, Any]:
    """Merge in-session profile tweaks into a YAML-ready mapping."""

    data: Dict[str, Any] = copy.deepcopy(base_yaml) if isinstance(base_yaml, dict) else {}

    data["name"] = profile_state.get("name")
    data["sex"] = str(profile_state.get("sex", "")).upper()
    data["age"] = int(profile_state.get("age")) if profile_state.get("age") is not None else None
    data["weight"] = float(profile_state.get("weight_kg", 0.0))
    data["weight_unit"] = "kg"

    if profile_state.get("height_cm") is not None:
        data["height_cm"] = profile_state.get("height_cm")

    data["hr_rest"] = profile_state.get("hr_rest")
    data["hr_max"] = profile_state.get("hr_max")
    data["hr_error_bpm"] = float(profile_state.get("hr_error_bpm", 3.0))

    cal_cfg = data.get("calories", {}) or {}
    cal_cfg["model_frac"] = float(profile_state.get("calories_model_frac", 0.2))
    cal_cfg["include_model_uncertainty"] = bool(
        profile_state.get("calories_include_model_uncertainty", True)
    )
    data["calories"] = cal_cfg

    script_cfg = data.get("script", {}) or {}
    if profile_state.get("recent_fit_files") is not None:
        script_cfg["recent_fit_files"] = int(profile_state.get("recent_fit_files", 5))
    fit_dir_val = profile_state.get("default_fit_dir")
    if fit_dir_val:
        script_cfg["default_fit_dir"] = str(fit_dir_val)
    data["script"] = script_cfg

    return data


def save_profile_to_yaml(
    profile_state: Dict[str, Any], profile_path: Path, base_yaml: Dict[str, Any]
) -> Dict[str, Any]:
    """Persist the profile to profile.yaml and return the saved YAML mapping."""

    yaml_data = build_profile_yaml(profile_state, base_yaml)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)
    print(f"Saved profile to {profile_path}")
    return yaml_data


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
    profile: Dict[str, Any],
    profile_yaml_path: Path,
    profile_yaml_data: Dict[str, Any],
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

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor("#f7f9fc")
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1.2])
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#f3f6fb")
    controls_ax = fig.add_subplot(gs[1])
    controls_ax.axis("off")

    # Keep mutable state for profile-driven updates
    profile_state = dict(profile)
    baseline_profile = dict(profile)
    profile_yaml_path = Path(profile_yaml_path)
    profile_yaml_data_current = copy.deepcopy(profile_yaml_data)
    zones_current = list(zones)
    zone_stats_current = list(zone_stats)
    kcal_estimate_current = kcal_estimate
    kcal_sigma_current = kcal_sigma

    # Zone shading
    zone_colors = [
        "#e0f3db",
        "#a8ddb5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
        "#fdae6b",
        "#fb6a4a",
    ]
    zone_patches: List[Any] = []

    def draw_zone_bands(zones_list: List[Dict[str, Any]]):
        nonlocal zone_patches
        for patch in zone_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        zone_patches = []

        for i, z in enumerate(zones_list):
            low = z.get("low", None)
            high = z.get("high", None)

            low_plot = y_full_min if low is None else low
            high_plot = y_full_max if high is None else high

            patch = ax.axhspan(
                low_plot,
                high_plot,
                facecolor=zone_colors[i % len(zone_colors)],
                alpha=0.25,
                zorder=0,
                label="_nolegend_",
            )
            zone_patches.append(patch)

    draw_zone_bands(zones_current)

    # Data
    ax.plot(
        minutes,
        hr_vals,
        alpha=0.35,
        linewidth=1.0,
        label="Raw HR",
        zorder=2,
        color="#2b8cbe",
    )
    ax.plot(
        minutes,
        smoothed_hr,
        linewidth=1.8,
        label="Smoothed HR",
        zorder=3,
        color="#fb6a4a",
    )
    ax.fill_between(
        minutes,
        lower,
        upper,
        alpha=0.22,
        label="±1σ (local)",
        zorder=1,
        color="#9ecae1",
    )

    ax.set_xlabel("Minutes from start")
    ax.set_ylabel("Heart rate [bpm]")
    ax.set_title("Heart rate vs time")

    ax.set_xlim(x_full_min, x_full_max)
    ax.set_ylim(y_full_min - y_margin, y_full_max + y_margin)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best")

    # Stats box (fixed in axes coords)
    def build_stats_lines(kcal_val, kcal_sigma_val):
        lines: List[str] = []
        if hr_stats is not None:
            mean_hr, median_hr, min_hr, max_hr = hr_stats
            lines.append(f"Mean:   {mean_hr:.1f} bpm")
            lines.append(f"Median: {median_hr:.1f} bpm")
            lines.append(f"Min:    {min_hr:.0f} bpm")
            lines.append(f"Max:    {max_hr:.0f} bpm")

        if kcal_val is not None:
            if kcal_sigma_val is not None and kcal_sigma_val > 0:
                lines.append(f"Est. cals: {kcal_val:.0f} ± {kcal_sigma_val:.0f} kcal")
            else:
                lines.append(f"Est. cals: {kcal_val:.0f} kcal")
        return lines

    stats_text_artist = None

    def update_stats_text(kcal_val, kcal_sigma_val):
        nonlocal stats_text_artist
        stats_lines = build_stats_lines(kcal_val, kcal_sigma_val)
        if stats_lines:
            stats_text = "\n".join(stats_lines)
            if stats_text_artist is None:
                stats_text_artist = ax.text(
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
            else:
                stats_text_artist.set_text(stats_text)
        elif stats_text_artist is not None:
            stats_text_artist.remove()
            stats_text_artist = None

    update_stats_text(kcal_estimate_current, kcal_sigma_current)

    # Dynamic zone labels on the right side (outside), updated on zoom
    zone_line_artists: List[Any] = []
    zone_text_artists: List[Any] = []
    zone_stats_version = 0
    last_zone_stats_version = -1
    last_ylim: Optional[Tuple[float, float]] = None

    def ensure_zone_artists(count: int):
        while len(zone_line_artists) < count:
            line = ax.plot(
                [1.0, 1.02],
                [0.0, 0.0],
                transform=ax.transAxes,
                linestyle="--",
                linewidth=0.8,
                color="k",
                zorder=5,
                clip_on=False,
            )[0]
            text = ax.text(
                1.03,
                0.0,
                "",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=8,
                zorder=5,
                clip_on=False,
            )
            zone_line_artists.append(line)
            zone_text_artists.append(text)

        while len(zone_line_artists) > count:
            zone_line_artists.pop().remove()
            zone_text_artists.pop().remove()

    def update_zone_labels(force: bool = False):
        nonlocal last_zone_stats_version, last_ylim

        if not zone_stats_current:
            for line, text in zip(zone_line_artists, zone_text_artists):
                line.set_visible(False)
                text.set_visible(False)
            return

        y_min, y_max = ax.get_ylim()
        if (
            not force
            and last_ylim is not None
            and last_ylim == (y_min, y_max)
            and last_zone_stats_version == zone_stats_version
        ):
            return

        last_ylim = (y_min, y_max)
        last_zone_stats_version = zone_stats_version

        span = max(y_max - y_min, 1e-6)

        ensure_zone_artists(len(zone_stats_current))

        for line, text, zstat in zip(zone_line_artists, zone_text_artists, zone_stats_current):
            low = zstat.get("low", None)
            high = zstat.get("high", None)

            low_plot = y_min if low is None else low
            high_plot = y_max if high is None else high

            if high_plot < y_min or low_plot > y_max:
                line.set_visible(False)
                text.set_visible(False)
                continue

            low_clip = max(low_plot, y_min)
            high_clip = min(high_plot, y_max)
            y_mid = 0.5 * (low_clip + high_clip)

            y_frac = (y_mid - y_min) / span
            y_frac = min(max(y_frac, 0.0), 1.0)

            label = f"{zstat['name']}: {zstat['time_min']:.1f} min ({zstat['pct']:.0f}%)"

            line.set_xdata([1.0, 1.02])
            line.set_ydata([y_frac, y_frac])
            line.set_visible(True)

            text.set_position((1.03, y_frac))
            text.set_text(label)
            text.set_visible(True)

        return

    update_zone_labels(force=True)

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

    last_hover_idx: Optional[int] = None

    def on_move(event):
        nonlocal last_hover_idx
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            last_hover_idx = None
            return

        x = event.xdata
        if x is None:
            return

        idx = find_nearest_index(x, minutes_arr)
        if last_hover_idx == idx and annot.get_visible():
            return

        last_hover_idx = idx
        update_annotation(idx)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # Left drag zoom (rectangle)
    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        ax.set_xlim(min(x1, x2), max(x1, x2), emit=False)
        ax.set_ylim(min(y1, y2), max(y1, y2), emit=False)
        update_zone_labels()
        fig.canvas.draw_idle()

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

    def clamp_to_full_range(xmin, xmax, ymin, ymax):
        new_xmin = max(x_full_min, xmin)
        new_xmax = min(x_full_max, xmax)
        new_ymin = max(y_full_min - y_margin, ymin)
        new_ymax = min(y_full_max + y_margin, ymax)
        return new_xmin, new_xmax, new_ymin, new_ymax

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

            new_xmin, new_xmax, new_ymin, new_ymax = clamp_to_full_range(
                x_center - span_x / 2,
                x_center + span_x / 2,
                y_center - span_y / 2,
                y_center + span_y / 2,
            )

            ax.set_xlim(new_xmin, new_xmax, emit=False)
            ax.set_ylim(new_ymin, new_ymax, emit=False)
            update_zone_labels()
            fig.canvas.draw_idle()

        elif event.button == 3:
            # Zoom out by factor 2 around click
            cur_xmin, cur_xmax = ax.get_xlim()
            cur_ymin, cur_ymax = ax.get_ylim()
            x_center = event.xdata if event.xdata is not None else 0.5 * (cur_xmin + cur_xmax)
            y_center = event.ydata if event.ydata is not None else 0.5 * (cur_ymin + cur_ymax)

            span_x = (cur_xmax - cur_xmin) * 2.0
            span_y = (cur_ymax - cur_ymin) * 2.0

            new_xmin, new_xmax, new_ymin, new_ymax = clamp_to_full_range(
                x_center - span_x / 2,
                x_center + span_x / 2,
                y_center - span_y / 2,
                y_center + span_y / 2,
            )

            ax.set_xlim(new_xmin, new_xmax, emit=False)
            ax.set_ylim(new_ymin, new_ymax, emit=False)
            update_zone_labels()
            fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes is not ax:
            return

        # Scroll up zooms in, down zooms out
        scale = 0.8 if event.step > 0 else 1.25
        cur_xmin, cur_xmax = ax.get_xlim()
        cur_ymin, cur_ymax = ax.get_ylim()
        x_center = event.xdata if event.xdata is not None else 0.5 * (cur_xmin + cur_xmax)
        y_center = event.ydata if event.ydata is not None else 0.5 * (cur_ymin + cur_ymax)

        span_x = (cur_xmax - cur_xmin) * scale
        span_y = (cur_ymax - cur_ymin) * scale

        new_xmin, new_xmax, new_ymin, new_ymax = clamp_to_full_range(
            x_center - span_x / 2,
            x_center + span_x / 2,
            y_center - span_y / 2,
            y_center + span_y / 2,
        )

        ax.set_xlim(new_xmin, new_xmax, emit=False)
        ax.set_ylim(new_ymin, new_ymax, emit=False)
        update_zone_labels()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    def on_close(event):
        plt.close("all")
        sys.exit(0)

    fig.canvas.mpl_connect("close_event", on_close)

    # Also update zone labels if toolbar zoom/pan changes limits
    ax.callbacks.connect("ylim_changed", lambda ax: update_zone_labels())
    ax.callbacks.connect("xlim_changed", lambda ax: update_zone_labels())

    # --- Profile controls under the plot ---
    controls_ax.set_title("Profile")

    field_specs = [
        ("Name", "name", True, str),
        ("Sex (M/F)", "sex", False, lambda v: str(v).strip().upper()),
        ("Age (years)", "age", False, int),
        ("Weight (kg)", "weight_kg", False, float),
        ("Rest HR", "hr_rest", True, float),
        ("Max HR", "hr_max", True, float),
        ("HR error bpm", "hr_error_bpm", False, float),
        ("Model frac", "calories_model_frac", False, float),
    ]

    text_boxes: Dict[str, TextBox] = {}
    tooltip_texts: Dict[Any, str] = {}

    def register_tooltip(ax_obj, message: str):
        tooltip_texts[ax_obj] = message
    ncols = 3
    nrows = (len(field_specs) + ncols - 1) // ncols
    start_x = 0.02
    usable_width = 0.96
    col_width = usable_width / ncols
    box_width = col_width - 0.025
    row_height = 0.18
    box_height = 0.15
    start_y = 0.84

    for idx, (label, key, allow_none, caster) in enumerate(field_specs):
        row = idx // ncols
        col = idx % ncols
        x0 = start_x + col * col_width
        y0 = start_y - row * row_height
        ax_box = controls_ax.inset_axes([x0, y0, box_width, box_height])

        initial_val = profile_state.get(key, "")
        if initial_val is None:
            initial_text = ""
        else:
            initial_text = f"{initial_val}"

        tb = TextBox(ax_box, label + "\n", initial=initial_text)
        text_boxes[key] = tb
        tb.label.set_fontsize(8.5)
        tb.text_disp.set_fontsize(9)
        register_tooltip(
            tb.ax,
            {
                "name": "Displayed name for your records.",
                "sex": "Biological sex (M/F) used by the calorie model.",
                "age": "Age in years used by the calorie estimate.",
                "weight_kg": "Body weight in kilograms for calorie math.",
                "hr_rest": "Resting heart rate in bpm.",
                "hr_max": "Estimated or measured maximum heart rate in bpm.",
                "hr_error_bpm": "Device noise floor (1σ) in bpm for uncertainty.",
                "calories_model_frac": "Fractional 1σ model error (e.g. 0.2 = 20%).",
            }[key],
        )

    fit_row = nrows
    fit_y = start_y - fit_row * row_height
    fit_button_width = box_width
    fit_button_height = 0.14
    fit_button_x = start_x + 2 * col_width

    fit_dir_display = controls_ax.text(
        start_x,
        fit_y + fit_button_height / 2,
        "",
        fontsize=9,
        ha="left",
        va="center",
    )

    def shorten_label(raw: str, max_len: int = 60) -> str:
        if len(raw) <= max_len:
            return raw
        keep = max_len - 1
        head = max(10, keep // 2)
        tail = keep - head
        return f"{raw[:head]}…{raw[-tail:]}"

    def update_fit_dir_display(dir_value: Optional[str]):
        label = dir_value if dir_value else "(not set)"
        fit_dir_display.set_text(f"FIT folder: {shorten_label(label)}")

    update_fit_dir_display(profile_state.get("default_fit_dir"))

    browse_ax = controls_ax.inset_axes(
        [fit_button_x, fit_y, fit_button_width, fit_button_height]
    )
    browse_button = Button(browse_ax, "Pick FIT folder")
    browse_button.label.set_fontsize(9)
    register_tooltip(
        browse_button.ax,
        "Choose the folder that holds your .fit files; saved as the default.",
    )

    include_unc_ax = controls_ax.inset_axes([0.02, 0.2, 0.15, 0.15])
    include_unc_checkbox = CheckButtons(
        include_unc_ax,
        ["Include model uncertainty"],
        [bool(profile_state.get("calories_include_model_uncertainty", True))],
    )
    for txt in include_unc_checkbox.labels:
        txt.set_fontsize(9)
    register_tooltip(
        include_unc_checkbox.ax,
        "Adds the model's fractional error on top of HR noise for calories.",
    )

    status_text = controls_ax.text(0.02, 0.05, "", fontsize=9)

    def on_pick_fit_folder(event=None):
        start_dir_raw = profile_state.get("default_fit_dir")
        start_dir = Path(start_dir_raw).expanduser() if start_dir_raw else Path.cwd()
        chosen_dir = select_directory_with_dialog(start_dir)
        if chosen_dir is None:
            status_text.set_text("Folder picker was cancelled.")
        else:
            profile_state["default_fit_dir"] = str(chosen_dir)
            update_fit_dir_display(profile_state["default_fit_dir"])
            status_text.set_text(f"Default FIT folder set to {chosen_dir}")
        fig.canvas.draw_idle()

    browse_button.on_clicked(on_pick_fit_folder)

    def parse_field(text: str, allow_none: bool, caster):
        cleaned = text.strip()
        if cleaned == "" and allow_none:
            return None
        return caster(cleaned)

    def collect_profile_from_fields() -> Dict[str, Any]:
        updated = dict(profile_state)
        for label, key, allow_none, caster in field_specs:
            text_val = text_boxes[key].text
            parsed = parse_field(text_val, allow_none, caster)
            if key == "sex" and parsed not in ("M", "F"):
                raise ValueError("Sex must be M or F")
            updated[key] = parsed

        updated["calories_include_model_uncertainty"] = include_unc_checkbox.get_status()[0]

        if updated.get("age") is None or updated.get("weight_kg") is None:
            raise ValueError("Age and weight are required.")

        return updated

    def apply_profile_changes(new_profile: Dict[str, Any]):
        nonlocal zones_current, zone_stats_current, kcal_estimate_current, kcal_sigma_current, zone_stats_version

        zones_updated = build_zones_from_config(
            new_profile.get("zones_cfg_raw"),
            new_profile.get("hr_rest"),
            new_profile.get("hr_max"),
        )
        zone_stats_updated, _ = compute_zone_stats(minutes, smoothed, zones_updated)
        kcal_estimate_updated, kcal_sigma_updated = estimate_calories_kcal_strict(
            minutes,
            smoothed,
            sigma,
            new_profile["sex"],
            int(new_profile["age"]),
            float(new_profile["weight_kg"]),
            float(new_profile["hr_error_bpm"]),
            model_frac=float(new_profile.get("calories_model_frac", 0.2)),
            include_model_uncertainty=bool(
                new_profile.get("calories_include_model_uncertainty", True)
            ),
        )

        zones_current = zones_updated
        zone_stats_current = zone_stats_updated
        zone_stats_version += 1
        kcal_estimate_current = kcal_estimate_updated
        kcal_sigma_current = kcal_sigma_updated

        draw_zone_bands(zones_current)
        update_zone_labels(force=True)
        update_stats_text(kcal_estimate_current, kcal_sigma_current)
        fig.canvas.draw_idle()

    def on_apply(event):
        nonlocal profile_state
        try:
            updated = collect_profile_from_fields()
            apply_profile_changes(updated)
            profile_state = updated
            status_text.set_text("Profile tweaks applied to plot.")
        except Exception as exc:
            status_text.set_text(f"Could not update: {exc}")
        fig.canvas.draw_idle()

    def on_save(event):
        nonlocal profile_state, baseline_profile, profile_yaml_data_current
        try:
            updated = collect_profile_from_fields()
            apply_profile_changes(updated)
            profile_state = updated
            saved_yaml = save_profile_to_yaml(updated, profile_yaml_path, profile_yaml_data_current)
            baseline_profile = dict(updated)
            profile_yaml_data_current = saved_yaml
            status_text.set_text("Profile saved to profile.yaml.")
        except Exception as exc:
            status_text.set_text(f"Could not save profile: {exc}")
        fig.canvas.draw_idle()

    def on_reset(event):
        nonlocal profile_state
        profile_state = dict(baseline_profile)
        for label, key, allow_none, caster in field_specs:
            val = profile_state.get(key, "")
            text_boxes[key].set_val("" if val is None else f"{val}")
        desired_unc = bool(
            profile_state.get("calories_include_model_uncertainty", True)
        )
        current_unc = include_unc_checkbox.get_status()[0]
        if desired_unc != current_unc:
            include_unc_checkbox.set_active(0)
        update_fit_dir_display(profile_state.get("default_fit_dir"))
        apply_profile_changes(profile_state)
        status_text.set_text("Reset to YAML values for this session.")

    save_ax = controls_ax.inset_axes([0.58, 0.05, 0.12, 0.14])
    save_button = Button(save_ax, "Save to file")
    save_button.label.set_fontsize(9)
    save_button.on_clicked(on_save)
    register_tooltip(
        save_button.ax,
        "Write the current session values back into profile.yaml.",
    )

    apply_ax = controls_ax.inset_axes([0.72, 0.05, 0.12, 0.14])
    apply_button = Button(apply_ax, "Apply changes")
    apply_button.label.set_fontsize(9)
    apply_button.on_clicked(on_apply)
    register_tooltip(
        apply_button.ax,
        "Recalculate zones and calories in-session without saving to disk.",
    )

    reset_ax = controls_ax.inset_axes([0.86, 0.05, 0.12, 0.14])
    reset_button = Button(reset_ax, "Reset")
    reset_button.label.set_fontsize(9)
    reset_button.on_clicked(on_reset)
    register_tooltip(
        reset_button.ax,
        "Restore the controls to the last saved YAML values.",
    )

    tooltip = fig.text(
        0,
        0,
        "",
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="#fff9c4", edgecolor="#fdd835", alpha=0.9),
        visible=False,
        zorder=10,
    )
    active_tip_ax: Optional[Any] = None

    def update_tooltip_position(event):
        fig_x, fig_y = fig.transFigure.inverted().transform((event.x, event.y))
        fig_x = min(max(fig_x + 0.01, 0.0), 0.98)
        fig_y = min(max(fig_y + 0.01, 0.0), 0.98)
        tooltip.set_position((fig_x, fig_y))

    def show_tooltip(ax_obj, event):
        message = tooltip_texts.get(ax_obj)
        if message is None:
            return
        if event.x is None or event.y is None:
            return
        update_tooltip_position(event)
        tooltip.set_text(message)
        tooltip.set_visible(True)

    def on_motion(event):
        nonlocal active_tip_ax
        ax_under = event.inaxes
        if ax_under in tooltip_texts:
            if event.x is None or event.y is None:
                return
            active_tip_ax = ax_under
            show_tooltip(ax_under, event)
            fig.canvas.draw_idle()
        elif active_tip_ax is not None:
            tooltip.set_visible(False)
            active_tip_ax = None
            fig.canvas.draw_idle()

    def on_axes_enter(event):
        nonlocal active_tip_ax
        ax_under = event.inaxes
        if ax_under in tooltip_texts and event.x is not None and event.y is not None:
            active_tip_ax = ax_under
            show_tooltip(ax_under, event)
            fig.canvas.draw_idle()

    def on_axes_leave(event):
        nonlocal active_tip_ax
        if active_tip_ax is not None:
            tooltip.set_visible(False)
            active_tip_ax = None
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("axes_enter_event", on_axes_enter)
    fig.canvas.mpl_connect("axes_leave_event", on_axes_leave)

    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    profile_path = Path.cwd() / "profile.yaml"
    profile, profile_yaml_data = get_profile(profile_path)

    fallback_dir = Path.cwd()
    fit_dir, profile, profile_yaml_data = ensure_fit_directory(
        profile,
        profile_yaml_data,
        profile_path,
        fallback_dir,
    )

    fit_path = choose_fit_in_folder(
        fit_dir,
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
        profile,
        profile_path,
        profile_yaml_data,
    )


if __name__ == "__main__":
    main()
