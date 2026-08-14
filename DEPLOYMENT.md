# Deploying to Render

Your Streamlit app lives at `UniversalQuantAgent/app/app.py` and is a
27-page multi-app (NBA, Finance, NFL analysis, and the three Fantasy pages:
Draft Room, Draft Assistant, and Season Tools). It depends on a second local
project, `fantasy_engine/`, for everything the Fantasy pages do. Both must be
deployed together.

## Before you do anything: one deviation from the ask

You asked for the start command `streamlit run streamlit_app.py ...`. I did
**not** create a `streamlit_app.py` wrapper file, because it would silently
break page navigation: Streamlit resolves each page's file (e.g.
`pages/25_Fantasy_Draft_Room.py`) relative to the *entry file's own folder*, and
`app.py` already owns that relationship with its `pages/` directory next to
it. A wrapper file living anywhere else would point navigation at the wrong
folder. The start command below targets the real entry file directly instead
— same result, without the risk.

## 1. Folder structure (what must exist, relative to the repo root)

```
AI Quant/                          <- repo root, push this whole thing
├── requirements.txt               <- NEW: installs both projects together
├── render.yaml                    <- NEW: Render Blueprint (optional but easiest)
├── .python-version                <- NEW: pins Python 3.12.7
├── .streamlit/
│   └── config.toml                <- NEW: production server settings
├── UniversalQuantAgent/
│   ├── requirements.txt           <- existing, unchanged
│   ├── app/
│   │   ├── app.py                 <- the real Streamlit entry point
│   │   ├── page_runtime.py
│   │   ├── fantasy_shared.py      <- setup/cards shared by the three Fantasy pages
│   │   └── pages/
│   │       ├── 25_Fantasy_Draft_Room.py
│   │       ├── 26_Fantasy_Draft_Assistant.py
│   │       ├── 27_Fantasy_Season_Tools.py
│   │       └── ... (24 more pages)
│   └── modules/                   <- existing, unchanged
└── fantasy_engine/
    ├── pyproject.toml             <- defines the `fantasy` package
    └── fantasy/                   <- what the Fantasy pages import
```

Nothing needs to move. The only new files are the five marked `NEW` above,
all at the repo root — already created.

## 2. requirements.txt (already created, full contents)

```txt
-r UniversalQuantAgent/requirements.txt
./fantasy_engine
```

The second line is the fix for the real bug this inspection found: without
it, `fantasy_engine` is never installed on Render and the Fantasy pages fail
with `ModuleNotFoundError: No module named 'fantasy'`.

## 3. Build command

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

## 4. Start command

```bash
streamlit run UniversalQuantAgent/app/app.py --server.port $PORT --server.address 0.0.0.0
```

## 5. Before you push: two things need to be committed for the first time

Checked directly — as of right now:

- **`UniversalQuantAgent/` has never been committed.** `git status` shows it
  entirely untracked.
- **`fantasy_engine/` has a large amount of uncommitted work**, including
  modules the Fantasy page currently imports (`room_brain`, `user_brain`,
  `draft_fusion`, `historical_adp`, and others). If you push without
  committing these, Render will deploy a stale, likely-broken version.

Both need to go in before deployment will work correctly.

## 6. Push to GitHub

```bash
git add UniversalQuantAgent fantasy_engine requirements.txt render.yaml .streamlit .python-version DEPLOYMENT.md
git commit -m "Add deployment config; commit app and engine for Render"
```

If you don't already have a GitHub repo for this:

```bash
gh repo create my-fantasy-app --private --source=. --remote=origin
```

(Omit `--private` for a public repo. If you'd rather not use the `gh` CLI,
create an empty repo at github.com/new instead, then run the two commands
below with the URL it gives you.)

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

I have not run any of the commands above — they're ready for you to run, or
tell me to run them and I will.

## 7. Deploy on Render

**Option A — Blueprint (uses the `render.yaml` already created):**

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New +** → **Blueprint**
2. Connect your GitHub account if you haven't, then select your repo
3. Render reads `render.yaml` and pre-fills everything — review and click **Apply**

**Option B — Manual web service (same result, if you skip the Blueprint):**

1. **New +** → **Web Service** → select your repo
2. **Root Directory**: leave blank (repo root)
3. **Runtime**: Python 3
4. **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
5. **Start Command**: `streamlit run UniversalQuantAgent/app/app.py --server.port $PORT --server.address 0.0.0.0`
6. **Instance Type**: Free (see the caveat below before relying on this for draft day)
7. Add an environment variable: `PYTHON_VERSION` = `3.12.7`
8. Click **Create Web Service**

First build installs pandas, plotly, scikit-learn, nba_api, nflreadpy (pulls
in polars), and the fantasy engine — expect the first deploy to take a few
minutes. Render gives you a live build log; watch for it to end with your
app listening on the assigned port.

No API keys or secrets are required — nothing in the app.py/Fantasy import
path reads `st.secrets` or environment variables today.

## 8. ⚠️ Free tier: read this before your draft

Render's **free** web services fall asleep after 15 minutes of no traffic,
and the *first* request after that takes 30–60+ seconds to wake back up. If
you open the link cold right as your draft starts, you could be staring at a
loading spinner at the worst possible moment.

**For draft day:** open the link yourself 5–10 minutes before your draft to
wake it up, then leave the tab open. If this draft matters enough that a
30–60s cold start is a real risk, Render's **Starter** plan ($7/month, no
sleep) removes this entirely — you can switch your service's instance type
in the Render dashboard at any time, including the morning of your draft.

## 9. Get the link onto your phone

Once deployed, Render shows your URL at the top of the service page —
something like `https://universal-quant-agent.onrender.com` (Render assigns
the subdomain from the service name; yours may differ slightly).

- **Email it to yourself:** copy the URL, send it in an email or a note-to-self
  message, open that email on your phone, tap the link. It'll open in your
  phone's browser — no app install needed.
- **Faster for draft day:** text the link to yourself instead of emailing —
  it'll be one tap away in Messages rather than buried in an inbox.
- Consider adding it to your phone's home screen (in Safari: Share → Add to
  Home Screen; in Chrome: ⋮ menu → Add to Home screen) so it opens like an app.
