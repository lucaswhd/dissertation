# v8 visual review, PDF pages 037–078 — final verification and review history

**Latest final identity:** PDF SHA256 `617b1c80a95c9650857949235a8b0ddbd1603b7c89ae2a872138d390146f8273`,78pages,1,884,499bytes. DOCX SHA256 `4a8be224061693504453aed26ef8e86e89b0d4a8ddbfbab290eb0b30e20c280a`. The Figure5.1 source-label clipping is now resolved; see the final focused addendum. Mathematical structures, captions and specified table layout checks pass within the recorded review scope. Minor inherited paragraph spacing remains; NTU template verification remains Pending. Earlier sections preserve prior hashes and findings as history.

## Round-1 history

Round1 predates native-math repair and subsequent rerenders; its page map is historical.

PDF: `/root/dissertation_v8_delivery/NTU_Dissertation_Reviewed_v8_review.pdf`

Round-1 SHA256: `10988e9de4bbc6521993ec8abdfff919d599feb674c08411b42330803844f2d6`; 77 PDF pages. All page numbers below are physical PDF indices, not printed chapter page numbers (for example PDF38 prints25).

## Coverage and method

- Visually inspected every page37–77 through four contact sheets: `contact_037_048.jpg`, `contact_049_060.jpg`, `contact_061_072.jpg`, `contact_073_077.jpg` in `/root/dissertation_local_audit/v8/pdf_qa/`.
- Independently enlarged **25** pages in that range: 37–42,46–48,50,52–58,70–77. This includes every main result Figure/Table, Figure3.6, both References pages, every appendix table and FigureA.1.
- Additionally enlarged declaration pages3–5 to inspect signature/date placeholders.
- Used PDF text extraction for reference-label completeness and word-boundary corroboration. All25 reference labels [1]–[25] remain; no extracted word lies outside the page rectangle in37–77. These checks supplement actual image inspection; they do not replace it.
- This assesses the supplied LibreOffice-rendered PDF. It does not guarantee identical Microsoft Word pagination or certify the official NTU declaration template.

## Findings needing attention

| Severity | Round-1 page(s) | Finding | Round-1 status / action |
|---|---|---|---|
| P1 |39–42,47–48 | Fractions, scripts and the Eq3.8 piecewise layout are inherited ordinary text, not native math. Eq3.6/3.7/3.10 numerator and denominator fall on separate lines without a fraction bar; Eq3.8 lacks proper cases; Eq3.9 lacks clear absolute-value structure; Eq4.1/4.2 are flattened | **Fail.** Helper `v8_math_repair.py` prepared separately at main's request; actual repaired rendering must be checked in round2 |
| P2 |46 and scientific notation elsewhere | Some `10−8`, `10−10`, learning-rate and tolerance exponents appear at baseline and can look like subtraction | **Needs formatting repair.** Native superscript conversion included in the helper; do not change values |
| P2 |57–58 | Short Table5.5 is split after three data rows. Only C20 radio plus repeated header appears on58 | Data complete; repeated header works. Prefer keep caption plus all4 rows together on58 if the reflow permits |
| P2 |76 | TableA.8 header `Median` wraps as `Media` / `n` | No data loss; widen column or minimally adjust header font. Main has taken this item |
| P2 |44,52 | A chapter-summary page contains little text and large whitespace;52 has34 extracted words | Not a blank page and no missing content. Optional pagination refinement; do not remove text just to shorten page count |
| P2 |Several text pages, including54–55 | Inherited manual paragraph/line fragmentation causes isolated short lines; e.g. `Small` at end of54, continuation on55; uneven word spacing is visible in other paragraphs | Readable, no overlap; polish if safe, without changing content or doing broad manuscript rewriting |

## Figure and table enlargement checks

| Object | PDF page(s) | Round-1 visual observation |
|---|---|---|
| Figure3.6 |38 | All3 panels complete. Segment-duration labels, log scales, next-action magnitude, raw/valid anchor counts and caption are visible. Caption and figure share a page; no crop/overlap observed |
| Table4.1 |46 | All7 setting rows complete; caption above table; rows inside margins |
| Table4.2 |50 | All protocol columns and5 setting rows visible; longer population/prompt cells wrap inside their cells |
| Table5.1 |53 | All4 segment rows and counts visible; no duplicated chart mixed into the table |
| Figure5.1 |54 | All4 category labels/bars and caption complete, on same page |
| Table5.2 |54 | All5 metric rows visible, ± values readable, no horizontal crop |
| Figure5.2 |55 | Pickup/radio panels, points, uncertainty bars, axes and caption complete; zero reference visible |
| Table5.3 |55 | All6 context/model rows and precision/recall/F1 columns visible; no row clipped |
| Figure5.3 event comparison |56 | Both task panels, F1/precision legend, error bars, axis labels and full caption visible; lower radio uncertainty bars are untruncated, as stated in caption. Figure and caption share page |
| Table5.4 |57 | All4 baseline rows and5 skill columns present; no crop |
| Table5.5 |57–58 | All4 data rows present; header repeats on58; one-row continuation is the layout concern above |
| Figure5.4 conditioning |58 | Both panels, n96/n12 labels, dashed reference, legend and full caption present; figure and caption share page |
| TableA.1 |72 |9 segment rows plus Round0 total present; no clipped bottom row |
| TableA.2 |72 |4 segment rows plus Round1 total present; kept before A.2 section heading |
| TableA.3 |72–73 | **31 data items plus header =32 logical rows complete.** Page72 contains first7 items, throughState input;73 continues24 items fromFine-tuning throughConditioning bootstrap resamples. `Parameter / Value` header repeats on73. No row split or cropped values observed |
| FigureA.1 |74 | All3 model loss panels, train/validation legend, axes and full caption visible; figure and caption share page |
| TableA.4 |74 | A20/B20/C20 configuration and experiment identifiers visible, within margins |
| TableA.5 |75 | All4 contexts andTotal652 visible; caption on same page |
| TableA.6 |75 | All4 contexts and model mean±SD cells present. Numeric cells wrap across2 lines but remain distinguishable and inside cell bounds |
| TableA.7 |76 | All6 predicted/TP rows visible; no crop |
| TableA.8 |76 | All8 model/context/metric rows andCI bounds present; headerMedian wraps awkwardly |
| TableA.9 |76 | B20/C20 and all4 perturbation columns visible; caption and table same page |
| References |70–71 | Entries[1]–[25] retained and readable; line wrapping is uneven in some entries, but no entry is cut off by page boundary/margin |
| Declaration placeholders (extra scope) |3–5 | Dotted date/signature areas and printed author name remain. Supervisor name/signature is a placeholder; no handwritten or image signature appears. This is a visible-placeholder check, not authentication or official-template certification |

## Page-by-page contact-sheet ledger

`C` = contact-sheet overview; `E` = additionally enlarged single-page image. Pass in this ledger means no gross crop/overlap/blank-page defect was observed at the stated inspection level. It does not override the specific mathematics and typography findings above.

| PDF page | Level | Visible content / observation |
|---:|---|---|
|37|C+E|Decision-sampling explanation and Figure3.6 introduction. Whitespace before figure moves to next page; no missing text/crop |
|38|C+E|Figure3.6 complete with caption; start of3.4 below; no overlap |
|39|C+E|Baseline definitions; **Eq3.6 mathematical-layoutFail** |
|40|C+E|Teacher-forced audit and event definitions; **Eq3.7 mathematical-layoutFail** |
|41|C+E|Conditioning protocol; **Eq3.8 mathematical-layoutFail** |
|42|C+E|Conditioning ratios and frustum method; **Eq3.9/3.10 mathematical-layoutFail** |
|43|C|Frustum/closed-loop methods; text inside page, no visible overlap |
|44|C|Methodology summary; sparse page, not blank |
|45|C|Chapter4/stack/training setup; no gross layout defect |
|46|C+E|Shared training and Table4.1 complete; baseline scientific-notation superscripts need repair |
|47|C+E|A/B/C setup; **Eq4.1 mathematical-layoutFail** |
|48|C+E|C normalisation/excludedD/frozen audit; **Eq4.2 mathematical-layoutFail** |
|49|C|Frozen audit/conditioning protocol; no gross layout defect |
|50|C+E|Table4.2 and geometric protocol; complete table, no crop |
|51|C|Closed-loop/QA protocol; no visible overlap |
|52|C+E|Short4.12 summary only; excessive whitespace, no missing-content evidence |
|53|C+E|Chapter5/Table5.1; complete |
|54|C+E|Figure5.1 and Table5.2; complete; short final line continues next page |
|55|C+E|Figure5.2 and Table5.3; complete |
|56|C+E|Event discussion and Figure5.3; complete with caption |
|57|C+E|Table5.4 and first3 rows ofTable5.5; short table continuation concern |
|58|C+E|Table5.5 final row with repeated header; Figure5.4 complete |
|59|C|Normalisation and historical rollout discussion; no gross layout defect |
|60|C|Reliability/results summary; no gross layout defect |
|61|C|Discussion/RQ1; no visible crop/overlap |
|62|C|RQ1/RQ2; no visible crop/overlap |
|63|C|RQ2/RQ3; no visible crop/overlap |
|64|C|RQ3/normalisation/MAE interpretation; no gross layout defect |
|65|C|Discussion/limitations; no gross layout defect |
|66|C|Limitations/summary; no gross layout defect |
|67|C|Chapter7/conclusion; no visible crop/overlap |
|68|C|RQ answers/future work; no visible crop/overlap |
|69|C|Future work/final remark; intentional end-of-chapter whitespace |
|70|C+E|References1–14; complete, no cropped text |
|71|C+E|References15–25; complete, no cropped text |
|72|C+E|TablesA.1/A.2 and first7 items ofA.3; complete visible rows |
|73|C+E|A.3 continued with repeated header and remaining24 items; no crop |
|74|C+E|FigureA.1/TableA.4 andA.3 manifest introduction; complete |
|75|C+E|TablesA.5/A.6; complete |
|76|C+E|TablesA.7/A.8/A.9; complete, Median header wraps |
|77|C+E|Historical rollout and reproducibility appendices; all text visible; end-of-document whitespace |

## Round-1 conclusion and required follow-up

In the reviewed41-page range there is **no observed image cropping, image/text overlap, lost appendix row, lost reference entry, genuinely blank page, or figure/caption page separation**. This is a finding about round1 at the stated inspection levels, not a claim of flawless typography.

Round1 **does not pass mathematical-layout QA**. Main is applying a source-matched native-OMML helper and will rerender; the resulting PDF will have a differenthash/pagination. Recheck affected equations, all figure/caption pages, Table5.5, TableA.3 continuation, TableA.8 header, References and placeholders against that new output before final visual sign-off. Do not reuse the page-map or Pass entries here as evidence that the later PDF has been checked.

## Round-2 candidate review — rejected numbering cache

Candidate PDF SHA256: `fccef0c6414001859264a1e5d1bb9299c59a36eefc4ccefb3b047ff465a13b07`,78pages. Independent contact sheets and page previews were extracted directly into`/root/dissertation_local_audit/v8/thesis_map_final_review/`; allpages37–78 were visually inspected again. Formula pages35–42/48, Table5.5 page58, TableA.3 pages73–74 and TableA.8 page77 were also enlarged.

- **Math displayPass in this candidate:** all12 native equations are visible, including5fractions, the two-branch cases, mean absolute difference and scripts. The intervening blank-formula render was caused by a missing LibreOfficeMath component; the parent restored it.
- **Table5.5 layoutPass:** all4data rows now stay together on58.
- **TableA.8 headerPass:** Median is no longer split intoMedia/n.
- **TableA.3 completenessPass:**5data items on73 and26on74 give31items total; Parameter/Value repeats on74.
- **NumberingFail:** after automaticfield updating, SEQfields count across chapters: figures5.7–5.10/A.11 and tables4.3–4.4/5.5–5.9/A.10–A.18 appear in captions andLoF/LoT. The literalbody references remain5.1–5.4/A.1 andthe originaltable numbers. This candidate is therefore **rejected as a final review version** even though images/math display correctly. Parent is correctingchapter resets andwill rerender.

This section records actual findings for thishash. A laterfinalrender must have its ownhash andfreshnumbering/pagechecks; nofinalPass is implied here.

## Prior verified round — corrected captions and Math-enabled PDF

**Reviewed final SHA256:** `dec5babcd7f9d16965d7baae210a5e09d76d92cc53cf60b938751bf186affeb0`.

**Size:** 1,843,057 bytes; **pages:** 78. Final coverage is PDF37–78 inclusive (42 pages), plus equation pages35–36. Hashing and preview extraction used the same immutable PDF byte snapshot.

Evidence: `/root/dissertation_local_audit/v8/thesis_map_final_verified/`. All42 final contact thumbnails were visually reviewed anew. Enlarged final pages actually viewed: **35,36,37,39,40,42,48,54,55,57,58,59,73,74,75,77**. Reference labels were independently checked by PDF text extraction. This final section supersedes earlier failed candidates.

### Final repaired-structure checks

| Check | Final evidence | Status |
|---|---|---|
| All12 display equations | 3.1–3.3 on35; 3.4 on36; 3.5 on37; 3.6 on39; 3.7 on40; 3.8–3.10 on42; 4.1–4.2 on48 | **Pass** |
| Mathematical structure | Five fraction bars, two cases with membership conditions, mean absolute difference, accents and scripts visibly rendered; no blank equations | **Pass** |
| Final editable DOCX | Independently counted12 native `m:oMath` objects and18 `w:tbl` tables | **Pass** |
| Ordered figure captions | 3.1–3.6, 5.1–5.4, A.1 on27/29/30/32/34/38/54/55/57/59/75 | **Pass** |
| Ordered table captions | 3.1/3.2, 4.1/4.2, 5.1–5.5, A.1–A.9; all18 checked in the actual final PDF | **Pass** |
| Main result panels and captions | Figures5.1–5.4 andA.1 remain complete and share pages with their captions. Event is5.3; conditioning is5.4 | **Pass for panel/caption integrity**, subject to the legacy axis-label caveat below |
| Table5.5 | Caption and all4 data rows together on58 | **Pass** |
| TableA.3 | First5 items on73, remaining26 on74; repeated header;31 items plus header =32 logical rows | **Pass** |
| TableA.8 | All8 rows and CI bounds visible on77; Median no longer splits | **Pass** |
| References | Ordered labels[1]–[25] retained across71–72 | **Pass** |
| Gross page defects | All42 contact pages checked; no missing page, overlapping panel or clipped data region observed | **Pass at stated inspection levels** |
| NTU declaration template | See separate official-template report | **Pending official template verification** |
| Microsoft Word equivalence | Only actual LibreOffice PDF and DOCX structure checked | **Not claimed** |

### Remaining minor typography limitations

1. **Legacy Figure5.1 ylabel clipping (P2).** On PDF54, the original880×440 PNG has its leftmost vertical ylabel partly cut at the image edge. Bars, percentages, tick values and category labels are intact. Direct extraction of the displayed image confirmed this is a source-asset issue, rather than Word cropping or exact paragraph line height. Evidence: `figure_5_1_actual_embedded.png` in the final-review directory. Re-exporting with a wider left bounding box would improve it. Do not claim that every label on every figure is completely uncropped.
2. Inherited manual paragraph fragmentation and uneven justified spacing remain in body/reference text. For example, `Small` occupies a short line on55, and some inline variable names remain flattened. The12 display equations themselves now have proper mathematical structure.
3. Summary/end-of-chapter pages44,61,70,78 retain substantial whitespace. They contain text and are not blank pages. Text was not removed to reduce page count.

### Final page-level ledger

C = final contact overview; E = additional final single-page enlargement. This ledger supersedes the round1 page map.

| PDF page | Level | Final observation |
|---:|---|---|
|37|C+E|Eq3.5 visible; sampling explanation and Figure3.6 introduction intact |
|38|C|All3 Figure3.6 panels and caption complete |
|39|C+E|Eq3.6 fraction visible; baseline text intact |
|40|C+E|Eq3.7 fraction visible; event definitions intact |
|41|C|Conditioning definitions continue; Eq3.8 moves to42 |
|42|C+E|Eq3.8 cases, 3.9 absolute values and3.10 fraction visible |
|43|C|Frustum/closed-loop methods; no overlap |
|44|C|Methodology summary; sparse but not blank |
|45|C|Chapter4 setup and training text intact |
|46|C|Training definitions; Table4.1 moves to47 |
|47|C|Table4.1 caption correct; all setting rows visible |
|48|C+E|Eq4.1/4.2 fractions and epsilon visible; no formula overlap |
|49|C|Final audit and conditioning protocols intact |
|50|C|Table4.2 caption correct; all columns visible |
|51|C|Geometry, closed-loop and QA text intact |
|52|C|Quality control and summary; no blank-page defect |
|53|C|Table5.1 caption correct; all segment rows visible |
|54|C+E|Figure5.1/Table5.2 correct; legacy ylabel clipping noted |
|55|C+E|Figure5.2 correct and complete; Table5.3 begins on56 |
|56|C|Table5.3 has all6 rows; event discussion intact |
|57|C+E|Figure5.3/Table5.4 correct; panels and captions complete |
|58|C+E|Table5.5 complete on one page |
|59|C+E|Figure5.4 conditioning panels and caption complete |
|60|C|Normalisation, historical rollout and reliability text intact |
|61|C|Short results summary; not blank |
|62|C|Discussion andRQ1 intact |
|63|C|RQ1/RQ2 intact |
|64|C|RQ2/RQ3 intact |
|65|C|RQ3, normalisation and MAE interpretation intact |
|66|C|Discussion and limitations intact |
|67|C|Limitations and summary intact |
|68|C|Chapter7 conclusion intact |
|69|C|RQ answers and future work intact |
|70|C|Future work and final remark intact |
|71|C|References1–14 retained |
|72|C|References15–25 retained |
|73|C+E|TablesA.1/A.2/A.3 numbered correctly; A.3 first5 items |
|74|C+E|A.3 remaining26 items and repeated header complete |
|75|C+E|FigureA.1/TableA.4 correct; training plot complete |
|76|C|TablesA.5/A.6 correct and complete |
|77|C+E|TablesA.7/A.8/A.9 correct; Median unbroken |
|78|C|Closed-loop and reproducibility appendix text complete |

**Final assessment:** Pass for the repaired mathematical structures, ordered captions, main panel/caption integrity and specified table completeness on this exact PDF. The minor source-label/paragraph-typography limitations and official-template Pending remain explicit. This is review-PDF validation, not official submission approval or a Microsoft Word rendering guarantee.

## Final focused addendum — Figure5.1 source-label issue resolved

Final PDF SHA256: `617b1c80a95c9650857949235a8b0ddbd1603b7c89ae2a872138d390146f8273`;78pages;1,884,499bytes. Final editable DOCX SHA256: `4a8be224061693504453aed26ef8e86e89b0d4a8ddbfbab290eb0b30e20c280a`. Both identities were verified against actual file bytes.

This was a focused comparison, not a repeated research audit. New contact sheets for all42pages37–78 were compared with the previous independently extracted final contacts. Of18 retained full-resolution page previews in that range, **only page54 changed**; the other17 are pixel-identical. Contact thumbnails also showed extremely small rasterisation differences on38/55/57/59/75, whose full-resolution previews are unchanged. No downstream pagination movement was found.

Page54 was enlarged directly from the new PDF at1.65×. The complete ylabel **Target centre in frustum (fraction)** is visible with adequate left margin; there is no clipped half-character. The four bars show81.7%,89.1%,100.0%,100.0%, with n=3,569/6,792/1,560/381, agreeing with the existing archived-count table and its displayed rounding. Fraction-valued y-axis ticks0–1 are consistent with the percentage annotations. Figure5.1 caption, Section5.2 heading and all5 Table5.2 rows remain complete and on the same page. No overlap or new clipping is visible.

Evidence: `/root/dissertation_local_audit/v8/thesis_map_figure51_final/focused_comparison.json`, new contact sheets and`page_054_fullsize.png`.

**Final status: Pass for this focused change.** The previous P2 Figure5.1 source-asset ylabel clipping is **resolved**, not a remaining action. Prior successful formula, caption-numbering and table-completeness checks carry forward with the unchanged full-resolution pages and layout. Remaining limitations are inherited paragraph/justification polish, **Pending official template verification**, and no guarantee of identical Microsoft Word rendering. No final Word/PDF was modified by this reviewer.
