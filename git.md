# Git Complete Guide - Step by Step Commands

## Table of Contents
- [Installation](#installation)
- [Initial Configuration](#initial-configuration)
- [Creating a Repository](#creating-a-repository)
- [Basic Commands](#basic-commands)
- [Branching and Merging](#branching-and-merging)
- [Remote Repository](#remote-repository)
- [Stashing](#stashing)
- [Viewing History](#viewing-history)
- [Undoing Changes](#undoing-changes)
- [Tags](#tags)
- [Git Ignore](#git-ignore)
- [Collaboration Workflow](#collaboration-workflow)
- [Advanced Commands](#advanced-commands)
- [Git Aliases](#git-aliases)
- [Best Practices](#best-practices)
- [Common Issues](#common-issues)

---

## Installation

### Windows
```bash
# Download from: https://git-scm.com/download/win
# Or use chocolatey
choco install git
```

### Mac
```bash
# Using Homebrew
brew install git
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install git
```

### Linux (Fedora)
```bash
sudo dnf install git
```

### Verify Installation
```bash
git --version
```

---

## Initial Configuration

### Set User Information
```bash
# Set your name
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Set default branch name to main
git config --global init.defaultBranch main

# Set default editor
git config --global core.editor "code --wait"  # for VS Code
git config --global core.editor "vim"          # for Vim
```

### View Configuration
```bash
# List all configurations
git config --list

# List global configurations
git config --global --list

# Check specific config
git config user.name
git config user.email
```

### Configure Line Endings
```bash
# For Windows
git config --global core.autocrlf true

# For Mac/Linux
git config --global core.autocrlf input
```

---

## Creating a Repository

### Initialize a New Repository
```bash
# Create a new directory
mkdir my-project
cd my-project

# Initialize git repository
git init

# Check status
git status
```

### Clone an Existing Repository
```bash
# Clone from URL
git clone https://github.com/username/repository.git

# Clone into specific folder
git clone https://github.com/username/repository.git my-folder

# Clone specific branch
git clone -b branch-name https://github.com/username/repository.git

# Clone with depth (shallow clone)
git clone --depth 1 https://github.com/username/repository.git
```

---

## Basic Commands

### Check Repository Status
```bash
# View status
git status

# Short status
git status -s
```

### Adding Files
```bash
# Add specific file
git add filename.txt

# Add all files in current directory
git add .

# Add all files in repository
git add -A

# Add files interactively
git add -i

# Add part of a file
git add -p filename.txt
```

### Committing Changes
```bash
# Commit with message
git commit -m "Your commit message"

# Commit with detailed message
git commit -m "Title" -m "Description"

# Add and commit in one step
git commit -am "Your commit message"

# Amend last commit
git commit --amend -m "Updated commit message"

# Amend without changing message
git commit --amend --no-edit
```

### Viewing Differences
```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged

# Show changes in specific file
git diff filename.txt

# Show changes between commits
git diff commit1 commit2

# Show changes between branches
git diff branch1 branch2
```

### Removing Files
```bash
# Remove file from working directory and staging
git rm filename.txt

# Remove file from staging only (keep in working directory)
git rm --cached filename.txt

# Remove directory
git rm -r directory-name
```

### Moving/Renaming Files
```bash
# Rename file
git mv old-name.txt new-name.txt

# Move file
git mv file.txt directory/
```

---

## Branching and Merging

### Creating Branches
```bash
# Create new branch
git branch feature-branch

# Create and switch to new branch
git checkout -b feature-branch

# Create branch from specific commit
git branch feature-branch commit-hash

# Create branch and switch (modern way)
git switch -c feature-branch
```

### Switching Branches
```bash
# Switch to existing branch
git checkout branch-name

# Switch to branch (modern way)
git switch branch-name

# Switch to previous branch
git checkout -

# Switch to main/master
git checkout main
```

### Listing Branches
```bash
# List local branches
git branch

# List all branches (local and remote)
git branch -a

# List remote branches
git branch -r

# List branches with last commit
git branch -v

# List merged branches
git branch --merged

# List unmerged branches
git branch --no-merged
```

### Merging Branches
```bash
# Merge branch into current branch
git merge feature-branch

# Merge with no fast-forward
git merge --no-ff feature-branch

# Merge and squash commits
git merge --squash feature-branch

# Abort merge
git merge --abort
```

### Deleting Branches
```bash
# Delete local branch (safe)
git branch -d branch-name

# Force delete local branch
git branch -D branch-name

# Delete remote branch
git push origin --delete branch-name

# Delete multiple branches
git branch -d branch1 branch2 branch3
```

### Rebasing
```bash
# Rebase current branch onto main
git rebase main

# Interactive rebase (last 3 commits)
git rebase -i HEAD~3

# Continue rebase after resolving conflicts
git rebase --continue

# Skip current commit
git rebase --skip

# Abort rebase
git rebase --abort
```

---

## Remote Repository

### Adding Remote
```bash
# Add remote repository
git remote add origin https://github.com/username/repository.git

# Add remote with different name
git remote add upstream https://github.com/original/repository.git
```

### Viewing Remotes
```bash
# List remotes
git remote

# List remotes with URLs
git remote -v

# Show remote details
git remote show origin
```

### Fetching and Pulling
```bash
# Fetch from remote
git fetch origin

# Fetch all remotes
git fetch --all

# Pull from remote (fetch + merge)
git pull origin main

# Pull with rebase
git pull --rebase origin main

# Pull from specific branch
git pull origin branch-name
```

### Pushing
```bash
# Push to remote
git push origin main

# Push all branches
git push --all origin

# Push tags
git push --tags

# Push and set upstream
git push -u origin main

# Force push (use with caution!)
git push --force origin main

# Force push with lease (safer)
git push --force-with-lease origin main
```

### Changing Remote URL
```bash
# Change remote URL
git remote set-url origin https://github.com/username/new-repository.git

# Verify change
git remote -v
```

### Removing Remote
```bash
# Remove remote
git remote remove origin
```

---

## Stashing

### Basic Stashing
```bash
# Stash current changes
git stash

# Stash with message
git stash save "Work in progress on feature"

# Stash including untracked files
git stash -u

# Stash including untracked and ignored files
git stash -a
```

### Viewing Stashes
```bash
# List all stashes
git stash list

# Show stash contents
git stash show

# Show stash contents with diff
git stash show -p
```

### Applying Stashes
```bash
# Apply most recent stash
git stash apply

# Apply specific stash
git stash apply stash@{2}

# Apply and remove stash (pop)
git stash pop

# Apply specific stash and remove
git stash pop stash@{2}
```

### Managing Stashes
```bash
# Create branch from stash
git stash branch branch-name

# Drop specific stash
git stash drop stash@{1}

# Clear all stashes
git stash clear
```

---

## Viewing History

### Log Commands
```bash
# View commit history
git log

# One line per commit
git log --oneline

# View with graph
git log --graph --oneline --all

# View last n commits
git log -n 5

# View commits by author
git log --author="John Doe"

# View commits with diffs
git log -p

# View commits in date range
git log --since="2 weeks ago"
git log --after="2024-01-01" --before="2024-12-31"

# View commits that modified a file
git log filename.txt

# View commits with specific message
git log --grep="bug fix"

# Pretty format
git log --pretty=format:"%h - %an, %ar : %s"
```

### Viewing Specific Commits
```bash
# Show commit details
git show commit-hash

# Show specific file in commit
git show commit-hash:filename.txt

# Show changes in last commit
git show HEAD
```

### Blame
```bash
# Show who changed each line
git blame filename.txt

# Show blame for specific lines
git blame -L 10,20 filename.txt
```

### Reference Log
```bash
# View reference log
git reflog

# View reflog for specific branch
git reflog show branch-name
```

---

## Undoing Changes

### Working Directory Changes
```bash
# Discard changes in file
git checkout -- filename.txt

# Discard all changes
git checkout -- .

# Restore file (modern way)
git restore filename.txt

# Restore all files
git restore .
```

### Staging Area Changes
```bash
# Unstage file
git reset HEAD filename.txt

# Unstage all files
git reset HEAD

# Unstage file (modern way)
git restore --staged filename.txt
```

### Commit Changes
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (unstage changes)
git reset HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo to specific commit
git reset --hard commit-hash

# Create new commit that undoes previous commit
git revert commit-hash

# Revert without committing
git revert -n commit-hash
```

### Clean Untracked Files
```bash
# Show what would be removed
git clean -n

# Remove untracked files
git clean -f

# Remove untracked files and directories
git clean -fd

# Remove untracked and ignored files
git clean -fdx
```

---

## Tags

### Creating Tags
```bash
# Create lightweight tag
git tag v1.0.0

# Create annotated tag
git tag -a v1.0.0 -m "Version 1.0.0"

# Tag specific commit
git tag -a v1.0.0 commit-hash -m "Version 1.0.0"
```

### Viewing Tags
```bash
# List all tags
git tag

# List tags with pattern
git tag -l "v1.*"

# Show tag details
git show v1.0.0
```

### Pushing Tags
```bash
# Push specific tag
git push origin v1.0.0

# Push all tags
git push origin --tags
```

### Deleting Tags
```bash
# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin --delete v1.0.0
```

### Checking Out Tags
```bash
# Checkout tag
git checkout v1.0.0

# Create branch from tag
git checkout -b branch-name v1.0.0
```

---

## Git Ignore

### Create .gitignore File
```bash
# Create .gitignore
touch .gitignore
```

### Common .gitignore Patterns
```gitignore
# Python
*.pyc
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Node
node_modules/
npm-debug.log
yarn-error.log
.npm

# Java
*.class
*.jar
*.war
target/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Logs
*.log
logs/

# Database
*.sqlite
*.db

# Compiled
*.exe
*.o
*.out

# Temporary
*.tmp
*.temp
tmp/
temp/

# Secrets
.env
*.key
*.pem
config/secrets.yml
```

### Ignore Already Tracked Files
```bash
# Remove from tracking but keep file
git rm --cached filename.txt

# Remove directory from tracking
git rm -r --cached directory/

# Commit the changes
git commit -m "Remove tracked files"
```

### Global .gitignore
```bash
# Create global gitignore
git config --global core.excludesfile ~/.gitignore_global

# Add patterns to global gitignore
echo ".DS_Store" >> ~/.gitignore_global
```

---

## Collaboration Workflow

### Fork and Clone Workflow
```bash
# 1. Fork repository on GitHub

# 2. Clone your fork
git clone https://github.com/your-username/repository.git

# 3. Add upstream remote
git remote add upstream https://github.com/original-owner/repository.git

# 4. Create feature branch
git checkout -b feature-branch

# 5. Make changes and commit
git add .
git commit -m "Add new feature"

# 6. Push to your fork
git push origin feature-branch

# 7. Create Pull Request on GitHub
```

### Keeping Fork Updated
```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream main into your main
git checkout main
git merge upstream/main

# Push updates to your fork
git push origin main
```

### Feature Branch Workflow
```bash
# 1. Update main branch
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/new-feature

# 3. Work on feature
git add .
git commit -m "Implement new feature"

# 4. Push feature branch
git push -u origin feature/new-feature

# 5. Create Pull Request

# 6. After merge, delete branch
git checkout main
git pull origin main
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

### Resolving Merge Conflicts
```bash
# 1. Attempt merge
git merge feature-branch

# 2. View conflicts
git status

# 3. Open conflicted files and resolve
# Look for conflict markers: <<<<<<<, =======, >>>>>>>

# 4. Mark as resolved
git add conflicted-file.txt

# 5. Complete merge
git commit -m "Resolve merge conflicts"
```

---

## Advanced Commands

### Cherry Pick
```bash
# Apply specific commit to current branch
git cherry-pick commit-hash

# Cherry pick multiple commits
git cherry-pick commit1 commit2 commit3

# Cherry pick without committing
git cherry-pick -n commit-hash
```

### Bisect (Find Bug Introduction)
```bash
# Start bisect
git bisect start

# Mark current commit as bad
git bisect bad

# Mark known good commit
git bisect good commit-hash

# Git will checkout commits for testing
# Mark each as good or bad
git bisect good
git bisect bad

# End bisect
git bisect reset
```

### Submodules
```bash
# Add submodule
git submodule add https://github.com/user/repo.git path/to/submodule

# Clone repository with submodules
git clone --recursive https://github.com/user/repo.git

# Initialize submodules after clone
git submodule init
git submodule update

# Update submodules
git submodule update --remote

# Remove submodule
git submodule deinit path/to/submodule
git rm path/to/submodule
```

### Worktrees
```bash
# Create worktree
git worktree add ../feature-branch feature-branch

# List worktrees
git worktree list

# Remove worktree
git worktree remove ../feature-branch
```

### Archive
```bash
# Create zip archive
git archive --format=zip --output=project.zip main

# Create tar archive
git archive --format=tar --output=project.tar main
```

---

## Git Aliases

### Create Aliases
```bash
# Common aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'
git config --global alias.amend 'commit --amend --no-edit'
```

### Useful Aliases
```bash
# Pretty log
git config --global alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

# Show branches
git config --global alias.branches "branch -a"

# Show remotes
git config --global alias.remotes "remote -v"

# Undo last commit
git config --global alias.undo "reset --soft HEAD~1"
```

---

## Best Practices

### Commit Messages
```bash
# Good commit message format:
# <type>: <subject>
#
# <body>
#
# <footer>

# Types: feat, fix, docs, style, refactor, test, chore

# Example:
git commit -m "feat: add user authentication

Implemented JWT-based authentication system
- Added login endpoint
- Added registration endpoint
- Added token validation middleware

Closes #123"
```

### Commit Guidelines
1. **Commit often** - Make small, logical commits
2. **Write clear messages** - Explain what and why, not how
3. **Use present tense** - "Add feature" not "Added feature"
4. **Keep commits focused** - One logical change per commit
5. **Test before committing** - Ensure code works

### Branch Naming
```bash
# Good branch names:
feature/user-authentication
bugfix/login-error
hotfix/security-patch
release/v1.2.0
docs/api-documentation

# Avoid:
fix
new-stuff
temp
```

### General Best Practices
1. **Pull before push** - Always pull latest changes first
2. **Don't commit sensitive data** - Use .gitignore for secrets
3. **Keep main/master stable** - Use feature branches
4. **Review before merging** - Use pull requests
5. **Tag releases** - Use semantic versioning
6. **Write README** - Document your project
7. **Use .gitignore** - Keep repository clean
8. **Backup regularly** - Push to remote often

---

## Common Issues

### Issue: Merge Conflicts
```bash
# View conflicts
git status

# Abort merge
git merge --abort

# After resolving manually
git add .
git commit
```

### Issue: Committed to Wrong Branch
```bash
# Move commit to new branch
git branch new-branch
git reset --hard HEAD~1
git checkout new-branch
```

### Issue: Need to Undo Last Push
```bash
# Revert on local
git reset --hard HEAD~1

# Force push (if branch not shared)
git push --force-with-lease origin branch-name
```

### Issue: Large File Committed
```bash
# Remove from last commit
git rm --cached large-file.zip
git commit --amend -m "Remove large file"

# Remove from history (use BFG Repo-Cleaner)
# Download from: https://rtyley.github.io/bfg-repo-cleaner/
```

### Issue: Accidentally Deleted Branch
```bash
# Find commit hash
git reflog

# Restore branch
git checkout -b branch-name commit-hash
```

### Issue: Wrong Commit Message
```bash
# Change last commit message
git commit --amend -m "Correct message"

# Change older commit message
git rebase -i HEAD~n
# Change 'pick' to 'reword' for commit
```

### Issue: Need to Split Commit
```bash
# Reset to previous commit
git reset HEAD~1

# Stage and commit separately
git add file1.txt
git commit -m "First logical change"

git add file2.txt
git commit -m "Second logical change"
```

---

## Useful Git Commands Reference

### Quick Reference
```bash
# Status and Info
git status                    # Check status
git log --oneline            # View history
git diff                     # View changes

# Basic Operations
git add .                    # Stage all
git commit -m "message"      # Commit
git push                     # Push to remote
git pull                     # Pull from remote

# Branching
git branch                   # List branches
git checkout -b new-branch   # Create and switch
git merge branch-name        # Merge branch

# Undo
git reset HEAD~1             # Undo last commit
git restore filename         # Discard changes
git revert commit-hash       # Revert commit

# Remote
git remote -v                # View remotes
git fetch                    # Fetch changes
git push origin main         # Push to remote
```

---

## Git Cheat Sheet

### Setup
```bash
git init                     # Initialize repository
git clone <url>              # Clone repository
git config --global user.name "Name"
git config --global user.email "email"
```

### Staging
```bash
git add <file>               # Stage file
git add .                    # Stage all
git reset <file>             # Unstage file
git rm <file>                # Delete file
```

### Committing
```bash
git commit -m "message"      # Commit with message
git commit -am "message"     # Add and commit
git commit --amend           # Amend last commit
```

### Branches
```bash
git branch                   # List branches
git branch <name>            # Create branch
git checkout <name>          # Switch branch
git merge <name>             # Merge branch
git branch -d <name>         # Delete branch
```

### Remote
```bash
git remote add origin <url>  # Add remote
git push origin <branch>     # Push branch
git pull origin <branch>     # Pull branch
git fetch                    # Fetch changes
```

### History
```bash
git log                      # View history
git log --oneline            # Compact log
git show <commit>            # Show commit
git diff                     # Show changes
```

---

## Resources

- [Official Git Documentation](https://git-scm.com/doc)
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [GitHub Guides](https://guides.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Learn Git Branching](https://learngitbranching.js.org/)
- [Oh My Git! (Game)](https://ohmygit.org/)

---

## License

This guide is provided as-is for educational purposes.

---

**Created by:** Your Name  
**Last Updated:** 2025-01-23  
**Version:** 1.0