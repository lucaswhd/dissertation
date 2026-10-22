# V8 document validation

日期：2026-10-22（Asia/Singapore）。本轮只修订v7，没有重新训练、仿真、数据集生成或重复完整研究审计。事件图原请求为Figure5.4；为保持首次出现顺序，正文用Figure5.3，原conditioning图顺延5.4。对应脚本stem保留原请求名，映射明确。

| Check | Status | Actual evidence |
|---|---|---|
| v7 preserved | Pass | SHA256 `e24f23f635b43b97297273ae1f85b66d97b2b09a39e971dee9725b76694afbb9`; local backup same hash |
| DOCX ZIP/XML and image relationships | Pass | V8_DOCX_STRUCTURE.json; no missing drawing links |
| Numbered figures / native tables | Pass | 11 figures, 18 native tables; exact chapter sequences checked in actual PDF |
| Table values / references | Pass | All 18 tables and numbered-reference text equal v7 |
| Images and captions | Pass | All11 complete and same-page in LibreOffice render; no srcRect crop/exact picture line-height |
| PDF all-page checks | Pass | 78 pages scanned, no text/image outside page, blank page, or caption without same-page image |
| Native mathematics | Pass | 12 OMML equations, fractions/cases restored from original definitions; actual rendered content checked |
| Page layout / indexes | Pass | A4,3 sections; cover no numbers, front Roman, body Arabic; 11/18 complete index entries |
| Official EEE/July2026 declaration | Pending | Pending official template verification; preserve real disclosure and signature placeholders |
| Microsoft Word rendering equivalence | Pending | PDF actually rendered with LibreOffice; final author Word preview still recommended |

DOCX SHA256 `4a8be224061693504453aed26ef8e86e89b0d4a8ddbfbab290eb0b30e20c280a`; PDF SHA256 `617b1c80a95c9650857949235a8b0ddbd1603b7c89ae2a872138d390146f8273`. This is a real review PDF, not an institutional submission certification. Font fallback: Times New Roman maps to Liberation Serif and mathematical text to STIXGeneral on this Linux renderer; Word fonts may lay out differently.
