# Operation

Once installed and configured:

1. A contributor opens or pushes to a PR in an opted-in repo.
2. GitHub delivers a `pull_request` webhook; the service verifies it and enqueues a job,
   cancelling any stale queued job for the same PR.
3. The worker gates the PR (opt-in + scope + caps), assembles the seed, runs the three lenses
   + synthesis, and posts one review with a verdict — REQUEST_CHANGES when a finding meets the
   repo's blocking threshold, otherwise COMMENT.
4. On a new push, the prior review is dismissed (REQUEST_CHANGES) or minimized (COMMENT) and a
   fresh review replaces it.

**Tuning a repo.** Edit `.github/heimdall.yml` to enable/disable or re-tune lenses, add custom
lenses, change the blocking threshold, narrow scope, or adjust the guardrail caps.

**Fork safety.** Config is read from the **base** branch ref by default. Only a same-repo PR
from a trusted author association (`OWNER` / `MEMBER` / `COLLABORATOR`) may have its **head**
config honored; a **fork PR is always forced to the base ref**, so a malicious fork can never
ship a config that disables the security lens, injects a lens prompt, or widens scope.
