# v8 native Word mathematical-layout repair

## Scope and root cause

Round-1 PDF (`10988e9de4bbc6521993ec8abdfff919d599feb674c08411b42330803844f2d6`, 77 pages) visually flattens fractions and the comparison-space piecewise formula. Both v7 and the pre-repair v8 DOCX contain **zero `m:oMath` objects**. Numerators, denominators and cases were inherited as ordinary text in consecutive paragraphs. Changing picture crop or font size cannot restore that mathematical structure.

Helper: `tools/plotting/v8_math_repair.py`.

API: `changes = repair_math(doc)`.

The helper edits only the supplied in-memory python-docx Document. It does not save, render, sign, or modify the final manuscript itself. Main must save a new v8 output and rerender. This is necessary mathematical formatting, not a new method or change in results.

## Original text, intended layout, and source basis

Paragraph indices below refer to the unchanged v7 (`e24f23f635b43b97297273ae1f85b66d97b2b09a39e971dee9725b76694afbb9`), not the shifting v8 indices. Exact original fragments are retained in the helper's `SOURCE` mapping and returned in each change record.

| Eq | Original v7 location / fragment | Native mathematical layout restored | Basis for meaning |
|---|---|---|---|
| 3.1 | p307, `s_t= bprev_t, qtorso_t, qL_t, gL_t, qR_t, gR_t` with scripts flattened | `s_t = (b_t^prev, q_t^torso, q_t^L, g_t^L, q_t^R, g_t^R)` | p314-equivalent state definition in §3.2.5; current v7 p306–309 explicitly enumerates the six components and dimensions. No transpose or extra component added |
| 3.2 | p311, `Δa(j) t+k = a(j) t+k − s(j) t` | `Δa_{t+k}^{(j)} = a_{t+k}^{(j)} − s_t^{(j)}` | p310: torso/arm action deltas are relative to the anchor state |
| 3.3 | p318, `(o_t,s_t,p_t) → (a_t+1,...,a_t+H)` | Correct subscripts and delimited tuples | p317/319–320: shift-1 future target; no change in horizon or alignment |
| 3.4 | p344, `Pdata(d=1)≈0.0396` | `P_data(d=1) ≈ 0.0396` | p343–345: natural exported fraction; original rounding preserved |
| 3.5 | p346, `Ptrain(d=1)=0.20` | `P_train(d=1) = 0.20` | p347: sampling target; not relabelled as observed draws |
| 3.6 | p355–356, numerator line `S=1−Em`, denominator line `E0` | `S = 1 − E_m / E_0`, with native fraction bar over `E_m` and `E_0` only | p353–354 explicitly defines model MAE `E_m` and no-motion MAE `E_0`; p357 says zero matches baseline. Therefore **not** `(1−E_m)/E_0` |
| 3.7 | p377–378, `Rmotion = E[|Δhat(a)^arm|]` and `E[|Δa^arm|]` | Native fraction of the two expected absolute arm deltas, with hat and `arm` superscript | p375–376/379–380: motion magnitude relative to expert, no direction-accuracy claim |
| 3.8 | p402–404: first branch, left-hand side, second branch stored out of normal reading order | `tilde(a)_x^(j) = { hat(a)_x^(j)−s_x^(j), j∈D; hat(a)_x^(j), j∉D }` | p401 defines D as 18 torso/arm dimensions; p405–406 says subtract the state actually supplied in each condition and leave other action groups absolute |
| 3.9 | p408–409, `D_i,x^(g)=mean tilde(a)_x^(g)−tilde(a)_ref^(g)` plus standalone period/number | `D_{i,x}^{(g)} = mean |tilde(a)_x^(g) − tilde(a)_ref^(g)|` | **p407 explicitly defines mean absolute difference over action horizon and dimensions.** Native absolute-value delimiters restore that stated definition; no L2 norm, new aggregation or per-source averaging added |
| 3.10 | p411–412, `Q_i,x^(g)=D_i,x^(g)` above `D_i,noise^(g)` | Native fraction `Q_{i,x}^{(g)} = D_{i,x}^{(g)} / D_{i,noise}^{(g)}` | p410 and p413–418: perturbation/noise ratio, per-anchor aggregation; not a ratio of pooled means |
| 4.1 | p481–482, `z(x)=x−μ` above `σ+ε` | `z(x) = (x−μ)/(σ+ε)` with native fraction | p480 and p484–486: A/B z-score transform, training-derived statistics |
| 4.2 | p483, `whereas C uses q(x)=2 x−q0.01 q0.99−q0.01+ε−1` | `q(x) = 2(x−q_0.01)/(q_0.99−q_0.01+ε) − 1` | Original order, percentile references and epsilon placement; p484–486 explains nominal [−1,1] scaling with no clipping. The English connector “whereas C uses” is retained |

All 12 equation numbers are retained exactly once. Surrounding definitions, numerical results, cross-references and all 18 native tables remain in place. Equations use OMML fractions, sub/superscripts, scalable delimiters, hat/tilde accents and a two-row piecewise matrix, with automatic line height so fractions/cases are not clipped by exact paragraph spacing.

## Scientific-notation typography

In ordinary body runs only, existing literals `10−5`, `10−6`, `10−7`, `10−8`, `10−10` (also ASCII/hyphen variants) are split into unchanged base `10` plus a native Word superscript exponent. Numeric values and signs do not change. This includes the already-written epsilon, AdamW settings, learning-rate values and floating-point tolerances. Runs with fields, tabs or non-text structures are not split. Native tables are not rewritten.

## Source safeguards and tests

- Preflight matches all 12 numbered equations and their exact original fragment sequences, normalising only Unicode mathematical alphabet variants and whitespace.
- If a fragment is absent or changed, the helper returns `status: Pending` and performs **no partial repair**. It does not infer a missing formula.
- Already native equations return `Unchanged`; repeated execution is idempotent.
- A targeted test against the actual v7 first failed because the helper was absent, then passed after implementation: 12 OMML equations, 5 native fractions, 2 cases rows, preserved numbers and 18 tables.
- A mismatched Eq3.6 denominator produced Pending with zero inserted equations.
- Save/reopen of the local self-test copy followed by a second repair produced no additional repairs; v7 source hash stayed unchanged.

Test command:

```bash
PYTHONPATH=/root/dissertation_local_audit/python_deps python /root/dissertation_local_audit/v8/test_math_repair.py
```

Local self-test artifact: `/root/dissertation_local_audit/v8/math_repair_selftest.docx`. This is not the final manuscript.

**Visual validation remains Pending until main saves the repaired v8 and rerenders it.** Native XML structure is evidence of a formatting repair, not a Microsoft Word rendering guarantee. Recheck all affected equation pages, downstream pagination, figure/caption pairing and the Table A.3 continuation after rerendering.

## Renderer-component diagnosis after integration

The first full rerender after OMML insertion displayed equation numbers but blank formula bodies. This was a **separate renderer-component failure**, not a successful mathematical render: the minimal LibreOffice installation lacked its Math component. Main restored the component and rerendered the **same** native-math probe without changing this helper.

Independent enlarged inspection of `/root/dissertation_local_audit/v8/native_math_probe.pdf` now confirms all12 equations visible: the5 intended fraction bars, two-branch cases withmembership conditions, mean absolute differences, accents and scripts. No formula-generation change was needed. The blank-formula PDF candidate was rejected. Full-document final-round page/hash validation is recorded separately in the visual-review report; the probe alone does not sign off that final PDF.

## Final integrated visual verification

The corrected78-page PDF, SHA256 `dec5babcd7f9d16965d7baae210a5e09d76d92cc53cf60b938751bf186affeb0`, was independently enlarged on equation pages35–37,39–40,42,48. All12 equations visibly preserve the intended fractions, cases, absolute values, accents and scripts. Final editable DOCX inspection also confirms12 native OMML objects and18 tables. **Math-layout verification: Pass for this final PDF.** Detailed page coverage and minor unrelated typography limitations remain in`V8_VISUAL_REVIEW_PAGES_037_077.md`.

## Final Figure5.1-only revision

Latest PDF SHA256 is `617b1c80a95c9650857949235a8b0ddbd1603b7c89ae2a872138d390146f8273`; final DOCX SHA256 is `4a8be224061693504453aed26ef8e86e89b0d4a8ddbfbab290eb0b30e20c280a`. Focused image comparison found only PDF54 changed among18 retained full-resolution pages37–78; the retained mathematical page images are unchanged. This revision resolves the unrelated legacy Figure5.1 ylabel clipping; it does not alter the restored formulas. See the visual report's final focused addendum.
