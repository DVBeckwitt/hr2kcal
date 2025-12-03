# hr-energy-lab

I use Strava and wanted a more accurate way to view my HR data and precise calorie burned number. This is written to do that with interactivity and error propogation.
---

## What this does

* Reads HR time series from `.fit` files (Garmin, Samsung, etc)
* Smooths noisy HR with an automatically chosen moving window
* Estimates calories with a transparent HR→kcal model
* Propagates HR noise and device error into a 1σ calorie error
* Computes time in HR zones and overlays zones on the plot
* Provides an interactive HR plot with hover tooltips and zoom, plus a console summary and CSV export


---

## Repository layout

```
hr-energy-lab/
├── pyproject.toml            # Packaging metadata (setuptools)
├── requirements.txt         # Runtime dependencies for quick installs
├── README.md                # This document
├── LICENSE                  # MIT License
├── run.bat                  # Windows helper that runs the module
├── fit_files/               # Default .fit directory with a sample file
├── examples/
│   └── profile.example.yaml # Sample profile configuration
└── src/hr_energy_lab/
    ├── __init__.py
    └── hr_interactive.py    # Main CLI / plotting tool
```

If you install the project with `pip install -e .` you will also get a console entrypoint called `hr-energy-lab` that launches the interactive tool.

---

## Features in more detail

### Input and preprocessing

* Loads `.fit` files using `fitparse` and extracts per-record timestamps and heart rate.
* Converts timestamps to minutes from start:
  \[
  t_i = \frac{t_i - t_0}{60\ \text{s}}
  \]

Limitations: only `.fit` is supported right now. TCX, GPX, or direct API integration would require extra parsers.

### HR smoothing and local noise estimate

Function: `compute_smoothed_auto(minutes, hr_vals, target_durations_min=None)`

* Chooses an odd window size (w) via leave-one-out cross validation over a set of target durations (default 0.1 to 2.0 min).
* For each candidate `w`, it minimizes the mean squared error between each point and the mean of its neighbors (LOOCV).
* Returns the smoothed HR, per-sample local standard deviation, chosen window, and effective duration.


### HR zones

Zones are built in `build_zones_from_config(zones_cfg, hr_rest, hr_max)`.

You can define zones in `profile.yaml` in two ways:

1. **Absolute bpm**

   ```yaml
   zones:
     - {name: Z1, low_bpm: 110, high_bpm: 130}
     - {name: Z2, low_bpm: 130, high_bpm: 150}
   ```

2. **As a fraction of heart rate reserve (HRR)** using `hr_rest` and `hr_max`:

   ```yaml
   hr_rest: 55
   hr_max: 190
   zones:
     - {name: Z1, low_pct: 0.50, high_pct: 0.60}
     - {name: Z2, low_pct: 0.60, high_pct: 0.70}
   ```

If no zones are specified but `hr_rest` and `hr_max` are known, a default 5-zone HRR scheme is used.


### Calorie estimation and uncertainty

Core function: `estimate_calories_kcal_strict(minutes, hr_mean, hr_sigma_local, sex, age, weight_kg, hr_error_bpm)`

1. Construct midpoint values between each pair of time samples, compute local and device-floor HR uncertainty, and derive an effective 1σ per midpoint.
2. Apply the Keytel-style HR→kcal formulas (per minute) for each midpoint, clipping negative rates to zero and integrating over time.
3. Propagate HR uncertainty via the derivative of the rate with respect to HR, and optionally combine with a fractional model error.

The console prints something like:

```
Calories (strict propagation + floor): E = 650 ± 40 kcal (hr_error_bpm=3.0)
```


### Interactive plotting

`plot_interactive(...)` creates an interactive matplotlib window with:

* HR vs time (raw, smoothed, ±1σ band)
* Colored HR zone bands in the background
* A stats box with HR metrics and calories ±1σ
* Dynamic zone labels that update when you zoom
* Hover tooltips and rectangle zoom + double-click/ right-click zoom controls

### In-plot profile tweaks

Below the figure, a profile panel mirrors common `profile.yaml` fields (name, sex, age, weight in kg, rest/max HR, HR error floor, calorie model fraction, model uncertainty toggle). Updating a field and clicking **Apply changes** recalculates calories, zones, and labels immediately. **Reset** reloads everything from `profile.yaml` for the current session.

### CSV export

`copy_csv_to_clipboard(minutes, hr_vals)` builds a CSV with headers and copies it to the Windows clipboard (or prints a notice elsewhere).

---

## Installation

Requirements:

* Python 3.9 or newer
* `numpy`, `matplotlib`, `fitparse`, `pyyaml`

Install for development (editable):

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -e .
```

Or install just the dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start

1. Place your `.fit` files in `fit_files/` (the default) alongside a `profile.yaml` or point the app at a different folder in the GUI.
2. Copy and edit `examples/profile.example.yaml` with your details.
3. From any folder, run:

   ```bash
   python -m hr_energy_lab.hr_interactive
   # or after installation
   hr-energy-lab
   ```
4. When prompted, point the script at the folder containing your `.fit` files (defaults to `fit_files/`). It lists the newest files there, parses HR vs time, chooses a smoothing window, computes HR stats and calories with uncertainty, copies CSV (Windows), and opens the interactive plot.
5. Use the profile panel under the plot to tweak values (weight, HR rest/max, error floor, model fraction, etc.) for quick what-if experiments without editing `profile.yaml`. Changes affect the live plot only; "Reset" brings you back to the YAML values.

### Sample FIT file

A demo activity (`sample.fit`) from the `fitparse` test suite is bundled in `fit_files/`. The app starts in that folder by default so new users can open the plot immediately before pointing the tool at their own data.

---

## profile.yaml format

```yaml
name: David
sex: M           # "M" or "F"
age: 32
weight: 86.0     # in kg unless you set weight_unit: lb
weight_unit: kg  # or "lb"
height_cm: 188   # optional

hr_rest: 55      # optional but needed for HRR based zones
hr_max: 190      # optional but needed for HRR based zones

# Device floor for 1 sigma HR error in bpm
# 3 bpm is a reasonable default for Galaxy Watch 4 class devices
hr_error_bpm: 3.0

# Optional custom zones
zones:
  - name: Z1
    low_pct: 0.50
    high_pct: 0.60
  - name: Z2
    low_pct: 0.60
    high_pct: 0.70
  - name: Z3
    low_pct: 0.70
    high_pct: 0.80
  - name: Z4
    low_pct: 0.80
    high_pct: 0.90
  - name: Z5
    low_pct: 0.90
    high_pct: 1.00
```

If `profile.yaml` is missing or invalid, the script falls back to a simple interactive prompt and uses a generic profile with no zones.

---

## Project goals and scope

This project aims to be:

* A clear, inspectable implementation of HR-based energy estimation
* A tool that quantifies how sensitive energy is to realistic HR noise
* A starting point for research or personal analysis, not a commercial product

It is not intended to:

* Replace medical-grade calorimetry
* Guarantee absolute calorie accuracy for weight management
* Provide long-term training load management on its own

---

## Possible extensions

If you want to grow this into a more competitive tool, obvious extensions include:

* Additional file formats (`.tcx`, `.csv` exports)
* A second uncertainty term for overall model error combined in quadrature with the strict HR error
* A small command line interface that writes HTML reports with plots and tables
* A calibration mode where you fit a simple scale factor to a trusted reference

---

## License

MIT – see [LICENSE](LICENSE).
