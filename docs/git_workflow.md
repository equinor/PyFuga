# Git Workflow Guide for PyFuga Contributors

This guide explains how to use Git for PyFuga development. It's written for people new to Git – no prior experience assumed. You'll learn the mental model, essential commands, and common workflows.

---

## Git Mental Model

Think of Git as a system that helps you:
1. **Track changes** to your code over time
2. **Save checkpoints** (commits) of working versions
3. **Work on separate branches** without affecting the main code
4. **Sync with your team** by pushing to and pulling from a remote repository

### Key Concepts

**Working Tree**  
Your actual files on disk – where you make edits.

```
┌─────────────────────┐
│   Working Tree      │  Your files (modified, added, deleted)
│  (your computer)    │
└─────────────────────┘
```

**Staging Area**  
A space between your working tree and commits. You choose which changes to keep via `git add` before committing.

```
┌─────────────────────┐
│   Working Tree      │
│                     │
│ ┌────────────────┐  │
│ │ Staging Area   │  │  Changes you've marked for commit
│ │ (git add)      │  │
│ └────────────────┘  │
└─────────────────────┘
```

**Commit**  
A saved checkpoint of your code at a point in time. Each commit has a message describing the change (e.g., "Add yaw LUT support").

```
┌─────────────────────┐
│   Working Tree      │
└─────────────────────┘
           ↓
   ┌──────────────┐
   │   Commits    │  Saved versions: [msg1] → [msg2] → [msg3]
   │   (History)  │
   └──────────────┘
```

**Branch**  
A separate line of development. Each branch has its own set of commits. The main branch is usually called `main` (or `master`); you create feature branches for new work.

```
main:    ○──○──○──○

feature: ○──○──○  (branches off from main)
```

**Remote**  
A copy of the repository on a server (e.g., GitLab). Allows collaboration and backup.

```
Your computer (local):         GitLab (remote):
┌──────────────┐              ┌──────────────┐
│  main branch │  ←→ sync ←→  │  main branch │
│ (your copy)  │              │  (shared)    │
└──────────────┘              └──────────────┘
```

---

## Essential Git Commands

### `git status`

**What it does:** Shows the current state of your working tree and staging area.

**Example output:**
```
On branch feature/my-changes
Your branch is 3 commits behind origin/feature/my-changes.

Changes not staged for commit:
  modified:   pyfuga/utils.py
  
Untracked files:
  test_file.py
```

**Understanding the output:**
- `origin` is the remote repository (e.g., GitLab). All your branches exist both locally (on your computer) and remotely.
- `origin/feature/my-changes` is your branch as it exists on the remote.
- "3 commits behind" means the remote has 3 commits you haven't pulled yet.

**When to use:** Before committing, before pushing, whenever you want to see what's changed.

**Command:**
```bash
git status
```

---

### `git fetch`

**What it does:** Downloads updates from the remote repository. Does NOT change your local files – just checks what's new on the remote.

**Example:**
```bash
git fetch origin
```

This checks if your branch has fallen behind the remote version.

---

### `git log --oneline --decorate --graph --all`

**What it does:** Shows a visual history of all commits on all branches.

**Example output:**
```
* 3a4b5c6 (HEAD -> feature/my-changes) Add yaw LUT support
* 7d8e9f0 Fix PreLUT indexing
| * 1a2b3c4 (origin/main) Update README
| * 5f6g7h8 Refactor ODE solver
|/
* 9i0j1k2 Initial commit
```

**Understanding the output:**
- `*` = a commit on the current branch
- `|` = a line showing the history of another branch
- `7d8e9f0` = the commit hash (unique identifier). Use the first 7 characters to reference a commit.
- `HEAD` = the current commit (where you are now)
- `HEAD -> feature/my-changes` = you're on the `feature/my-changes` branch
- `(origin/main)` = where the `main` branch is on the remote
- The graph structure visually shows how branches relate: your `feature/my-changes` branched off from somewhere, while `origin/main` continued on its own path

**Tip:** Install the GitLens extension in VS Code (recommended in [.vscode/extensions.json](.vscode/extensions.json)) to see this information in a cleaner, interactive interface.

**When to use:** To understand the commit history and what others have done. The graph shows which commits belong to which branches.

**Command:**
```bash
git log --oneline --decorate --graph --all
```

---

### `git diff`

**What it does:** Shows the exact changes (line by line) between two versions.

**Examples:**

See changes in your current working tree (not yet staged):
```bash
git diff
```

See changes you've staged (ready to commit):
```bash
git diff --staged
```

See changes between your branch and the remote:
```bash
git diff origin/feature/my-changes
```

See changes in a specific file:
```bash
git diff pyfuga/utils.py
```

**When to use:** Before committing, to review what you changed. Or to see what the remote version looks like compared to yours.

**Tip:** VS Code has a built-in Git diff view. Right-click a file and select "Compare with Git History" or "Open Change".

---

### `git rebase @{u}`

**What it does:** Integrates changes from the remote branch into your local branch by re-applying your commits on top of the remote's latest commit.

**Why use it?** Keeps the commit history clean and linear (instead of creating merge commits).

**Mental model:**
```
Before rebase:
  local:   ○──○──○ (your work)
  remote:  ○──○────○──○ (new remote commits)

After rebase:
  local:   ○──○──○──○──────○──○
                   ↑      ↑
                   (what was here)
```

Your commits are re-applied on top of the remote's latest.

**Command:**
```bash
git rebase @{u}
```

`@{u}` means "the upstream" – the remote branch your local branch tracks.

**What to do if rebase fails:** See "Resolving Conflicts" below.

---

### `git merge @{u}`

**What it does:** Integrates changes from the remote by creating a merge commit (a commit that combines two branches).

**Why use it?** Safer than rebase; preserves full history. Use this if rebase fails or if you're not comfortable with rebasing.

**Command:**
```bash
git merge @{u}
```

---

### `git stash`

**What it does:** Temporarily saves your changes (both staged and unstaged) so you can have a clean working tree.

**Example:**
```bash
git stash
```

Your changes are saved. You can then `git rebase` or switch branches safely.

**To restore:**
```bash
git stash pop
```

---

## Daily Workflow

Follow this pattern every day:

### 1. Start Your Session

Before you begin, sync with the remote:

```bash
git fetch
git status
```

Check if your branch is behind:
- If you see `Your branch is X commits behind origin/...`, you need to sync.

### 2. Sync with Remote (if behind)

If your branch is behind:

```bash
git rebase @{u}
```

This brings your branch up to date.

**If you have unsaved changes:** Stash them first:
```bash
git stash
git rebase @{u}
git stash pop
```

**If rebase fails:** See "Resolving Conflicts" below.

### 3. Make Your Changes

Edit files, create new features, fix bugs.

Check your progress:
```bash
git status
```

See the exact changes:
```bash
git diff
```

### 4. Commit Your Changes (after testing)

Stage your changes:
```bash
git add pyfuga/utils.py  # or git add . for all changes
```

Commit with a descriptive message:
```bash
git commit -m "feature: add yaw LUT interpolation support"
```

Use the same message style as CONTRIBUTING.md describes.

**Run tests before committing:**
```bash
python scripts/dev.py test
```

Or with Pixi:
```bash
pixi run test
```

### 5. Before Pushing

Make sure everything passes locally:

```bash
python scripts/dev.py fmt        # Auto-format code
python scripts/dev.py check-fmt  # Verify formatting
python scripts/dev.py test       # Run tests
```

Or with Pixi:
```bash
pixi run ci
```

### 6. Push to Remote

Once all checks pass:

```bash
git push
```

Your changes are now on the remote, and you can open a Merge Request (MR).

---

## Visual Workflow (VS Code)

VS Code has built-in Git support that helps visualise these steps:

- **Source Control panel** (left sidebar, Git icon): Shows modified files, staged changes, commit interface
- **Diff view**: Right-click a file → "Compare with Git History" to see changes
- **Git Graph** (if you install GitLens extension): Visual representation of `git log --oneline --decorate --graph --all`

---

## Common Scenarios

### Scenario 1: "I see 'Your branch is behind origin'"

**What this means:** The remote has commits you don't have yet.

**Solution:**

```bash
git fetch                   # Check what's on remote
git log --oneline --decorate --graph --all  # See the difference
git rebase @{u}             # Integrate remote commits into your branch
```

---

### Scenario 2: "My branch has diverged from remote"

**What this means:** You made commits while the remote also got new commits. Your histories are different.

**Solution:**

Same as above:
```bash
git rebase @{u}
```

This re-applies your commits on top of the remote's latest.

---

### Scenario 3: "I made changes but forgot to sync from remote first"

**What this means:** You edited files and made commits, but the remote has moved forward. You need to sync without losing your changes.

**Solution:**

```bash
git stash                  # Temporarily save your changes
git rebase @{u}            # Sync with remote
git stash pop              # Restore your changes
python scripts/dev.py test # Test to make sure everything works
```

---

### Scenario 4: "I want to see what changed between my version and the original"

**What this means:** You want to compare your current branch with `main`, or with the remote version.

**Solution (Command line):**

```bash
git diff main              # Compare with main branch
git diff origin/main       # Compare with remote main
git diff HEAD~1            # Compare with your last commit
```

**Solution (VS Code):**

1. Right-click a file in the Source Control panel
2. Select "Compare with Git History"
3. Choose the commit/branch to compare against

---

### Scenario 5: "I made a copy of a file to see the original"

**What this means:** You might have done something like `cp utils.py utils.py.bak` to keep the original around while editing.

**Solution:**

You don't need the copy! Git has a full history. Use:

```bash
git diff utils.py          # See your changes
git log --oneline utils.py  # See all past versions of this file
git show <commit>:utils.py  # View the file at a specific commit
```

Or in VS Code, right-click the file and "Compare with Git History".

---

## Resolving Conflicts

If `git rebase @{u}` fails with a conflict, it means you and the remote both changed the same lines.

**You'll see this in your files:**

```python
<<<<<<< HEAD
# Your version
def new_function():
    return 42
=======
# Remote version
def new_function():
    return 24
>>>>>>> origin/feature/my-changes
```

**To fix in VS Code (easiest):**

VS Code highlights conflicts visually with interactive buttons:
1. Open the conflicted file – you'll see **Accept Current Change**, **Accept Incoming Change**, and **Accept Both Changes** buttons above each conflict
2. Click the button for what you want (or manually edit the file)
3. Manually verify the result makes sense (especially if you chose "Accept Both")
4. Stage the fixed file: Click the `+` button in the Source Control panel, or run `git add utils.py`
5. Continue the rebase: Run `git rebase --continue` in the terminal

**To fix manually (command line):**

1. Open the conflicted file(s) in your editor
2. Decide which version to keep (or combine both)
3. Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
4. Stage the fixed file: `git add utils.py`
5. Continue the rebase: `git rebase --continue`

**If it's too complicated:**

Start over:
```bash
git rebase --abort
git merge @{u}
```

Merge creates a merge commit instead, which preserves full history but is simpler to resolve.

---

## Troubleshooting CI Failures

When CI fails, the error message tells you what went wrong. Here's how to debug locally:

### Formatting failed

**Error in CI:** "Black formatting check failed" or "Ruff linting failed"

**Fix locally:**
```bash
python scripts/dev.py fmt
```

This auto-fixes most formatting issues. Commit the changes and push again.

**Debug:** To see what's wrong without fixing:
```bash
python scripts/dev.py check-fmt
```

---

### Tests failed

**Error in CI:** "pytest failed" or "Some tests errored"

**Fix locally:**
```bash
python scripts/dev.py test
```

This runs the full test suite. You'll see which tests failed and why.

**Debug specific test:**
```bash
python scripts/dev.py test -k test_name  # Run a specific test
python scripts/dev.py test -v            # Verbose output
```

If tests pass locally but fail in CI, it might be:
- Python version mismatch (check `python --version`)
- Missing dependency (check your environment)
- Platform-specific issue (Linux vs Windows vs macOS)

For help, see the documentation build section in [CONTRIBUTING.md](../CONTRIBUTING.md).

---

### Documentation build failed

**Error in CI:** "Sphinx build failed" or "Documentation build errored"

**Fix locally:**
```bash
python scripts/dev.py build-docs
```

Check the error message carefully. It usually points to the file and line with the problem.

Common issues:
- Broken link in `.rst` file
- Missing reference (e.g., `:class:\MyClass\` but `MyClass` doesn't exist)
- Incorrect reStructuredText syntax

---

## Tips & Tricks

### See a pretty log

```bash
git log --oneline --decorate --graph --all
```

Alias it for easier access:
```bash
git config --global alias.lg "log --oneline --decorate --graph --all"
git lg  # Now you can use this shorthand
```

---

### Undo your last commit (keep changes)

```bash
git reset --soft HEAD~1
```

Your changes are back in the staging area but not committed.

---

### See branches

List all branches (local and remote):
```bash
git branch -a
```

Create a new branch:
```bash
git switch -c feature/my-new-feature
```

---

### Delete a local branch (after merging)

```bash
git branch -d feature/old-feature
```

---

## Summary: One-Command Reference

| Task | Command |
|------|---------|
| Check status | `git status` |
| See changes | `git diff` |
| See history | `git log --oneline --decorate --graph --all` |
| Sync with remote | `git fetch` then `git rebase @{u}` |
| Stage changes | `git add <file>` |
| Commit | `git commit -m "<message>"` |
| Push | `git push` |
| Stash changes | `git stash` |
| Restore stash | `git stash pop` |

---

## Further Reading

- [Official Git documentation](https://git-scm.com/doc)
- [GitHub's Git Handbook](https://guides.github.com/introduction/git-handbook/) (concepts apply to all Git hosting)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)

---

**Questions?** Ask a colleague or create an issue in the PyFuga repository.
