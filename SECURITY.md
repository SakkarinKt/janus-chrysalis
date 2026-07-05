# Security Policy

`janus-chrysalis` is an early-stage, pre-release research project (Phase 1). It
ships no released software yet, but we still take security and responsible
disclosure seriously.

## Supported versions

| Version            | Supported          |
| ------------------ | ------------------ |
| `main` (latest)    | :white_check_mark: |
| any earlier commit | :x:                |

Only the current `main` branch is maintained. There are no tagged releases yet.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public
issue for a security problem.

- **Preferred:** use GitHub's
  [Private vulnerability reporting](https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  on this repository (the **Security → Report a vulnerability** button).
- **Alternative:** email the maintainer at <sakkarink12@gmail.com> with
  enough detail to reproduce the issue.

Please include: a description, steps to reproduce (or a proof of concept), the
affected file(s)/commit, and the potential impact as you see it.

## What to expect

As a solo-maintained project, response times are best-effort:

- **Acknowledgement:** within ~7 days.
- **Assessment & fix plan:** communicated once the report is triaged.
- **Disclosure:** coordinated with you; we'll credit reporters who wish to be
  named once a fix (or mitigation) is available.

## Scope

In scope: source code and configuration in this repository, and the daily
"loop" automation contract (`loop/GOAL.md`). Out of scope: third-party
dependencies (report those upstream), and issues requiring privileged access to
a maintainer's account or machine.

## Good-faith safe harbor

We will not pursue or support legal action against researchers who act in good
faith, avoid privacy violations and service disruption, and give us reasonable
time to remediate before public disclosure.
