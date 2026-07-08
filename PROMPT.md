# Kickoff prompt

Paste this into Claude Code from the repo root (with `CLAUDE.md` and `SPEC.md` present):

---

Build the `lifesim` project described in `SPEC.md`, following every non-negotiable
in `CLAUDE.md`. Work in the phases listed at the bottom of `CLAUDE.md`: after each
phase, run the tests for that phase and show me the output before continuing. Do
not collapse the multi-objective outcome into a single scalar utility anywhere.
Do not add third-party dependencies to the core — matplotlib and any scenario
loaders must be optional and import-guarded. When you finish Phase 1, run
`python3 -m lifesim --domain all --u 0.3 --seed 1` and paste the result so I can
sanity-check the dynamics before you build the rest.

---

If you'd rather build it incrementally yourself, ask Claude Code for one phase at a
time: "Do Phase 1 only, then stop."
