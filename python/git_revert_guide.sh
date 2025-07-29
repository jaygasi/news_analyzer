# 🔄 REVERTING TO THE TAGGED VERSION (Future use)

# Option 1: Create a new branch from the tag (RECOMMENDED)
# This lets you test the old version without losing current work
git checkout -b revert-to-stable v1.2.0-stable-high-volume

# Option 2: Hard reset to the tag (⚠️ DESTRUCTIVE - commits current changes first!)
git add . && git commit -m "WIP: saving before revert"  # Save current work
git reset --hard v1.2.0-stable-high-volume             # Go back to tagged version

# Option 3: Just look at the tagged version temporarily
git checkout v1.2.0-stable-high-volume  # Read-only view
git checkout main                        # Return to current version

# 📋 USEFUL TAG COMMANDS

# List all tags
git tag -l

# See tag details
git show v1.2.0-stable-high-volume

# Delete a tag (if needed)
git tag -d v1.2.0-stable-high-volume           # Local
git push origin --delete v1.2.0-stable-high-volume  # Remote

# Compare current code with tagged version
git diff v1.2.0-stable-high-volume..HEAD