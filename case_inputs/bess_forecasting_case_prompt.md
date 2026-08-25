# Prompt: BESS Peak-Shaving Load Forecasting Case Study

## Role
You are acting as a senior data scientist / quant analyst at a company that operates stationary
Battery Energy Storage Systems (BESS) for Commercial & Industrial (C&I) customers in Germany.
I'm giving you a self-contained technical case study to complete end-to-end. Treat the output as
a real deliverable a colleague could open, run, and extend without needing to ask me anything.

I have roughly 3-4 hours of effective work budgeted for this, so prioritization is itself part of
what's being evaluated. I want a **complete, working, well-reasoned solution**, not an exhaustive
one. Simple and interpretable beats sophisticated and opaque unless you can justify the added
complexity.

## Business context
We operate stationary BESS for C&I customers in Germany. A key use case is **peak shaving**:
reducing load peaks to lower demand charges, which can represent 30-50% of a C&I customer's
electricity bill. A short-term consumption forecast feeds directly into:
- Charge/discharge scheduling
- Peak shaving threshold decisions
- Operational optimization of the battery

**The core challenge is asymmetric error cost.** Under-forecasting a peak means missing the
chance to shave it -> a real, billed demand charge. Over-forecasting wastes battery capacity
that could have been used to arbitrage market prices instead. A good forecast (and a good
choice of metric) has to reflect this asymmetry, not just minimize a symmetric error like RMSE.

**Objective:** build a simple, robust, interpretable load forecast for a single C&I customer site.

This case evaluates: Python coding skills, data cleaning/preprocessing, time-series fundamentals,
building a pragmatic baseline model, and interpretation of results in an energy-economics context.

## How I want you to work
Before writing any code:
1. Give me a short (2-3 sentence) restatement of your understanding of the objective and what's
   being evaluated, so I can confirm we're aligned before you commit time to an approach.
2. Give me a brief bullet-point plan: the steps you'll take, and — given the time-box — what
   you'll prioritize vs. deliberately cut or simplify, with a one-line reason for each cut.

Then work through Part 1 -> Part 2 -> Part 3 below, in order. **After each part, pause and give
me a short progress update**: what you did, the key findings or decisions, and what's next —
then continue on your own initiative. Don't stop to ask for approval unless something is
genuinely ambiguous or blocking (e.g. the data doesn't match the assumed schema); in that case,
state your best assumption, flag it clearly, and keep moving rather than stalling on it.

## Data
[Attach the time series file here before sending this prompt.] Don't assume its structure —
actually inspect it first: columns, dtypes, timestamp format and timezone, sampling frequency,
length/date range, missing timestamps or gaps, duplicates, obvious outliers or sensor artifacts,
and whether the unit (kW vs kWh, instantaneous vs. interval-averaged) makes physical sense.

---

## Part 1 — Data Understanding, Preparation & EDA
- Explore the series and assess data quality (gaps, duplicates, outliers, resolution, DST
  handling, plausibility of values).
- Prepare/clean the data, and document *why* for each non-trivial choice (e.g. how you handle
  gaps — interpolate vs. flag vs. drop — and why that's defensible for a peak-shaving use case
  specifically, where filling over a real peak could hide the most important information).
- Analyze load patterns relevant to peak shaving: daily/weekly seasonality, peak timing and
  magnitude, peak persistence/duration, load factor, weekday vs. weekend behavior, any visible
  trend.
- Surface insights an operations or commercial person would actually act on — not just
  descriptive stats.

**Deliverables:** a prepared dataset ready for modeling; brief documentation of data issues,
assumptions, and preprocessing choices; a small set of the most informative visualizations
(not a dashboard — curate); a short written summary of the main insights and their relevance
to peak shaving.

## Part 2 — Forecasting Approach
- Design and implement a forecasting approach suitable for supporting peak-shaving decisions.
  Favor a pragmatic, interpretable baseline (e.g. seasonal-naive / persistence-with-seasonality,
  or a simple regression on calendar + lag features) over something exotic — but justify the
  choice explicitly against 1-2 alternatives you considered and decided against.
- Define a validation strategy appropriate for time series (e.g. time-based / walk-forward
  split — no random k-fold on temporal data) and explain why.
- Choose evaluation metric(s) deliberately in light of the asymmetric cost structure described
  above — e.g. don't rely on symmetric RMSE/MAE alone; consider something like a
  peak-focused or asymmetric loss (pinball loss / quantile-style scoring, or a simple
  cost-weighted metric that penalizes under-forecasted peaks harder than over-forecasts), and
  explain the reasoning even if you ultimately keep it simple.
- Explicitly discuss forecast horizon: what horizon(s) actually matter for charge/discharge
  scheduling and threshold decisions (e.g. next 15 min to next few hours vs. day-ahead), how
  horizon choice trades off against achievable accuracy, and what you'd recommend and why.

**Deliverables:** explanation of the modeling choice/approach, validation strategy, and metric(s);
a working baseline model with results; a discussion of what an appropriate forecasting horizon
would be for this use case.

## Part 3 — Business Interpretation & Reflection
- Interpret the results specifically from a peak-shaving lens: how would this forecast actually
  inform charge/discharge scheduling and threshold-setting in practice? Where does the model's
  error profile create real financial risk (missed peaks) vs. just lost optimization upside?
- Reflect concisely on limitations, risks (including what happens operationally when the model
  is wrong in each direction), and what additional data or improvements would matter most in a
  production setting (e.g. weather, calendar/holiday info, customer process schedules, submetering,
  price signals) — prioritized, not just listed.

**Deliverables:** discussion of how the forecast informs peak-shaving decisions; a concise
reflection on limitations, risks, and highest-value next improvements.

---

## Output format & code quality
- A notebook that documents the exploration and can double as a presentation of the work
  (narrative + code + visuals + short written takeaways at each stage, not just code cells).
- A clear entry point (a `main.py` script or a top-level function) so a colleague can run the
  whole pipeline without reading the notebook first.
- Code organized so a colleague could read, run, and extend it without asking me questions —
  reasonable structure (e.g. a small `src/` module for data prep and modeling, notebook for
  exploration/narrative, a short README, a requirements list). This is not meant to be
  production-hardened; I'm judging structure, clarity, and judgment, not deployment-readiness.
- Keep dependencies to a sensible core stack (pandas, numpy, matplotlib/seaborn or plotly,
  and statsmodels/scikit-learn/lightgbm only if actually justified) — don't over-engineer.

## Before you tell me you're done
Self-check the solution against the case's actual evaluation criteria: Python coding skills,
data cleaning/preprocessing, time-series fundamentals, a pragmatic baseline model, and
interpretation in an energy-economics context. Flag anything you deliberately scoped out due
to the time budget, and what you'd do next with more time.
