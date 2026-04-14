# ACES B2AI — Pediatric Voice Hackathon

Team ACES competing in the [2026 Voice AI Symposium Hackathon](https://www.eventsquid.com/event.cfm?id=29517) (May 6, 2026 · St. Petersburg, FL)

## Research Question
What acoustic features in pediatric voice recordings are associated with specific pathological conditions, and can ML models reliably identify these acoustic correlates?

## Dataset
Bridge2AI-Voice Pediatric Dataset v1.0.0  
PhysioNet DOI: [10.13026/y7mp-eh56](https://physionet.org/content/b2ai-voice-pediatric/1.0.0/)  
300 participants · Ages 2–18 · 22,620 recordings · 5 North American sites

> ⚠️ Data access requires PhysioNet credentialing + signing the Bridge2AI Voice DUA.  
> See `docs/data_access.md` for instructions. **Never commit data to this repo.**

## Team
| Name | Role |
|------|------|
| Akshay | Project Lead |
| Pranav | ML Engineer |
| Michael | Data Engineer |

## Repo Structure
aces-b2ai/
├── notebooks/          # Analysis notebooks (run in order)
├── src/aces_b2ai/      # Custom utilities
├── results/figures/    # Output figures
├── docs/               # Data access instructions
├── data/               # LOCAL ONLY - never committed
└── environment.yml     # Python environment

## Setup
```bash
conda env create -f environment.yml
conda activate aces-b2ai
```

## Citation
Bensoussan Y, et al. (2025). Bridge2AI-Voice Pediatric Dataset (version 1.0.0). PhysioNet. DOI: 10.13026/y7mp-eh56