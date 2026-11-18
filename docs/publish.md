# Existing public repository

Repository: https://github.com/KosTyanka/toxic-comment-detection

This is the owner's existing PUBLIC repository. Publish only the clean source, SQL, tests, documentation, aggregated results, report and presentation. Keep the private laptop runtime, comment text, databases, trained models, wheels, installers and credentials outside this repository.

Before future updates, inspect the local and remote history:

```bash
gh auth status
git remote -v
git fetch origin
git log --oneline --graph --all -10
git status --short
git diff --check
git ls-files -ci --exclude-standard
```

Preserve remote history for routine updates and integrate compatible remote changes before pushing. Rewriting history requires explicit owner authorization; use an exact expected-head lease for any authorized replacement.

After an authorized push, verify the remote commit, files and PUBLIC visibility:

```bash
git push origin main
gh repo view KosTyanka/toxic-comment-detection --json url,visibility,defaultBranchRef
```

The public source checkout deliberately omits data and learned artifacts. The separate private Windows laptop package supplies these for offline examination and must not be uploaded here.
