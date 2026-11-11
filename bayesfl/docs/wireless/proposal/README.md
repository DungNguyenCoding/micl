# Wireless FedKLS: IEEE-style LaTeX manuscript section

## Open in Overleaf

1. Upload this ZIP as a new project.
2. Select `main.tex` as the main file.
3. Select **pdfLaTeX**. Recompile normally; references are embedded, so BibTeX is not required.

The preview uses the installed `IEEEtran` journal class in two-column format. It is an IEEE-style draft, not a claim of compliance with every journal's submission-specific instructions. No custom fonts, images, or proprietary dependencies are included. Replace the placeholder author line before submission.

## Files

- `main.tex`: standalone preview wrapper.
- `preamble.tex`: required packages and macros.
- `wireless_model.tex`: orthogonal digital channel, SNR/power convention, payloads, and synchronized timing.
- `joint_problem.tex`: posterior reconstruction objective and joint/reduced optimization.
- `method1.tex`: Lagrangian decomposition, local KL threshold, dense comparison, price search, pseudocode, and bounds.
- `implementation.tex`: implementation modules, safeguards, baselines, and validation workflow.
- `references.tex`: numbered bibliography.
- `paper_section.tex`: one-section drop-in version of the complete body, with its internal headings demoted by one level.
- `IMPLEMENTATION.md`: suggested function interfaces and integration order; not a full training implementation.
- `CHANGELOG.md`: substantive differences from the earlier long investigation note.

## Insert into an existing manuscript

Copy `paper_section.tex`, merge the relevant packages/macros from `preamble.tex`, and add `\input{paper_section}` where the section belongs. Merge the four bibliography entries from `references.tex` into the manuscript's own bibliography rather than introducing a second References section. The drop-in file contains no document class, title, abstract, bibliography, or `document` environment. Its label prefix is `fkl:` to reduce collisions with the parent manuscript; inspect any macro/theorem-name collisions in the parent preamble.

Alternatively, keep the four numbered sections by importing the separate body files. Those use the original label names.

## Modeling decisions preserved

- All participating clients train full local models.
- Current-round importance is available before allocation; clients wait for the slowest local preparation.
- Packet headers, control/grant time, server computation, and aggregation latency are omitted analytically; the bitmap remains.
- A common prescribed round target defines a common upload interval.
- Version B selects **any integer** count from 1 through d, then assigns B_k=L_k(m_k)/(U*c_k).
- Orthogonal digital frequency resources are assumed, **not a particular OFDM waveform**.
- The current derivations use bandwidth-independent spectral efficiencies. The explicit physical example uses **fixed power spectral density**, not fixed total transmit power.
- Only Method 1 is developed. There is no MILP or other proposed solver in this draft.
- The practical method returns a feasible incumbent and a dual-gap diagnostic/certificate where justified. Global optimality is not assumed.

## Scope

The document gives a mathematical formulation, proposed allocation method, and implementation plan. No new wireless FedKLS training results are reported. General optimization references support standard dual/KKT concepts; the FedKLS-specific derivations are presented as the proposed specialization.
