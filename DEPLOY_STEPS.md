# Commit, push & redeploy (backend data fix)

## 1) Commit and push the new `backend/data/` folder

From your **backend** directory (this is your Git repo root):

```bash
cd /Users/nakulpednekar/Cursor_projects/NRL_Tryscorers/backend

# Stage the data folder and both CSV files
git add data/

# Commit with a short message
git commit -m "Add data folder with tryscorers CSV for Railway deploy"

# Push to your remote (usually origin/main)
git push origin main
```

If your default branch has a different name (e.g. `master`), use that instead of `main` in the last command.

---

## 2) Redeploy the backend on Railway

### Option A: Automatic redeploy (recommended)

If your Railway project is linked to this repo (e.g. the **backend** repo on GitHub/GitLab):

- Pushing to the linked branch (e.g. `main`) **triggers a new deploy automatically**.
- After `git push origin main`, open the [Railway dashboard](https://railway.app/dashboard), select your backend project, and check the **Deployments** tab. You should see a new deployment building and then going live (usually within a few minutes).

### Option B: Manual redeploy

If you didn’t push, or want to redeploy without a new commit:

1. Go to [railway.app](https://railway.app) and sign in.
2. Open the project that hosts **nrltryscorersbackend-production** (or your backend service).
3. Click the **backend service**.
4. Open the **Deployments** tab.
5. Click the **⋮** (or “Redeploy”) on the latest deployment and choose **Redeploy** (or use “Deploy” from the main service view if your plan shows that).

After the new deployment is **Live**, try again:

- `GET https://nrltryscorersbackend-production.up.railway.app/api/players`

You should get a successful response and the app should load player data.
