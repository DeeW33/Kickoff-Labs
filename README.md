# Football predictor website

A GitHub Action runs the model on a schedule, writes JSON, and publishes `site/index.html` plus that JSON to GitHub Pages. No server needed.

## Setup (about 10 minutes)

1. Create a new GitHub repo and push this folder to the `main` branch.
2. Repo **Settings > Pages > Build and deployment > Source: GitHub Actions**.
3. For college football, get a free key at https://collegefootballdata.com/key, then add it under **Settings > Secrets and variables > Actions > New repository secret** named `CFBD_API_KEY`. Without it the site deploys with NFL only.
4. Open the **Actions** tab, choose "Update predictions and deploy site", and click **Run workflow**. The first run takes 10 to 20 minutes while data downloads; later runs reuse a cache.
5. Your site appears at `https://<your-username>.github.io/<repo-name>/`.

## Schedule
Runs daily at 11:00 UTC and on Saturday and Sunday at 15:00 UTC (edit the `cron` lines in `.github/workflows/update.yml`). GitHub pauses scheduled runs on repos with no activity for 60 days; re-enable them from the Actions tab.

## Run locally
    pip install -r requirements.txt
    python football_predictor.py export --league nfl --out site/data/nfl.json
    cd site && python -m http.server 8000     # open http://localhost:8000

## Notes
- Keep the footer attribution and disclaimer. Open-Meteo's free tier is for non-commercial use.
- If you monetize the site, check betting-content rules for ads and affiliates where you operate.
