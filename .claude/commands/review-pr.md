---
description: Review a GitHub PR with summary and interactive feedback
argument-hint: <pr-number>
---

You're reviewing PR #$ARGUMENTS in this repo.

Step 1: Run `gh pr view $ARGUMENTS` and `gh pr diff $ARGUMENTS`.
Summarise what the PR does, why it exists, and the overall shape
of the change. Keep this to ~5 sentences.

Step 2: Walk through the diff file-by-file. For each non-trivial
change, explain the mechanism — what it does and why the author
likely chose this approach. Read surrounding code with the Read
tool when you need wider context; the diff alone is rarely enough.

Step 3: List concerns in priority order:
- Correctness bugs (highest)
- Edge cases and error paths
- API/contract changes that affect callers
- Performance or resource issues
- Style/idiom nits (lowest, only if notable)

Step 4: For each concern, ask me whether to: leave it, draft a
PR comment, or fix directly on the branch. Wait for my answer
before acting.

Do NOT post anything to GitHub until I explicitly say so.