# Repeated checks of an unchanged Mind

`npm run eval:repeats` presses Check this Mind again and again on Minds nobody edits, with a real chat model and the app's real search, and counts what each sentence gets each time (#1608). A User who checks the same text twice expects the same verdicts, and never "Nothing in your Documents says this" for a sentence a Document states. Neither `npm test` nor CI runs it.

## What it does

1. A temporary data folder, as for the other evaluations, with the Library of a first run: the example's Documents (`createExamples`) beside the set's own agreement, so the search weighs the agreement against other Documents as it does for a newcomer. Documents are added with embeddings off and searched with the built-in reranking model, as Users have them by default. The interface is in the set's language.
2. For each set, a Mind with its sentences, one paragraph each.
3. Check this Mind, as the button does on an unchanged Mind: the first time the model names the claims and checks the new sentences ("changed"), every later time it checks them all ("all"). `INCARNAMIND_EVAL_REPEATS` times (10 by default).
4. After each check, every sentence's result is read from its mark, as the editor reads it, and named by how its card shows it:
   - **supported**, **contradicted**;
   - **partly**: unsupported, and the card says what the Documents support and what they don't, quoting the Passage the check read;
   - **differs**: unsupported, and the card quotes a line that says something different;
   - **nothing**: unsupported, and the card says nothing in the Documents says it;
   - **not checked**, with why.
5. The report: per set and model, each sentence's outcomes and how many checks gave its most frequent verdict (partly, differs and nothing are one verdict, unsupported), false Supported, "nothing" on a sentence the Documents state, how many checks gave other verdicts than the check before, and every result that isn't the expected verdict in the check's own words. It goes into `eval/results/<start time>-repeats/` (`report.md`, `report.json`).

The cases are in `cases.ts`; `repeat.ts` checks any Mind's sentences in any core, so other sets can use it with their own Documents.

## The sets

An employment agreement in English and a 劳动合同 in Chinese (two-page PDFs, invented for this evaluation), and four Minds:

| Set | Sentences |
|---|---|
| `en`, `zh` | The three sentences of #1608 (notice as a paraphrase; holiday with a wrong figure; "Staff" for "the Employee"), the agreement's own words, a car allowance it doesn't mention, and a near miss that names days it doesn't |
| `en-control`, `zh-control` | The home-working clause four ways: a paraphrase about the same Employee, the Employee by name, "Staff" (员工) and "Employees" (职工) |

## Running it

Setting the variables is the consent to send the sentences and Passages to the model:

```sh
INCARNAMIND_EVAL_CHAT_KIND=anthropic \
INCARNAMIND_EVAL_CHAT_MODEL=claude-haiku-5-5 \
INCARNAMIND_EVAL_CHAT_KEY=sk-ant-... \
npm run eval:repeats
```

- `INCARNAMIND_EVAL_REPEATS`: checks of each Mind (10).
- `INCARNAMIND_EVAL_REPEATS_SETS`: only these sets, by id (`en,zh`).
- `INCARNAMIND_EVAL_REPEATS_BUDGET`: stops after the check that takes the spend past this many dollars, at the app's own count (5).

**Cost and time:** ten checks of the four Minds take about 5 minutes and $0.10 on Claude Haiku 5.5, and about 9 minutes and $2.20 on Claude Sonnet 5.5.

## Results

### 2026-10-22: before and after the partly-supported card for a sentence with no Citation

Before: `main` at bf47b4631, measured with this evaluation's first draft on the same Documents, Library and search (no `zh-control` yet). After: the change that keeps the Passage a check quoted for the part it covers and shows the partly-supported card. The verdicts are the model's in both; the change is what an Unsupported card says. Counts are of 10 checks; S supported, C contradicted, U unsupported (partly / nothing / differs after).

| Set · sentence | Haiku 5.5 before | Haiku 5.5 after | Sonnet 5.5 before | Sonnet 5.5 after |
|---|---|---|---|---|
| en · staff-home | 10 S | 9 S, 1 U nothing | 5 S, 5 U nothing | 8 S, 2 U partly |
| en · home-days (near miss) | 5 U nothing, 5 C | 4 U differs, 6 C | 10 U nothing | 10 U partly |
| en · car (absent) | 10 U nothing | 9 U nothing, 1 U partly | 10 U nothing | 10 U nothing |
| en-control · staff-home | not run | 10 S | 8 S, 2 U nothing | 9 S, 1 U partly |
| zh · staff-home | 9 S, 1 U nothing | 9 S, 1 U partly | 9 S, 1 U nothing | 10 S |
| zh · home-days (near miss) | 10 U nothing | 10 U partly | 10 U nothing | 10 U partly |
| zh · car (absent) | 10 U nothing | 9 U nothing, 1 U partly | 10 U nothing | 9 U nothing, 1 U partly |
| zh-control · staff-home | not run | 9 S, 1 U partly | not run | 8 S, 2 U partly |
| zh-control · employees-home | not run | 9 S, 1 U differs | not run | 10 S |

Every other sentence gave the same verdict in all its checks, before and after: notice, the paraphrases and the agreement's own words Supported, the wrong holiday figure Contradicted. No check gave a false Supported.

- **"Nothing in your Documents says this" on a sentence the agreement states:** 9 checks before (Sonnet 8, Haiku 1), each with the clause quoted in the check's own reason; 1 after, with two more sets run (Haiku, English, a reply that named no covered part).
- **What still varies:** whether "Staff" (员工) counts as the agreement's Employee. Sonnet 5.5 said no in 2 of 10 English checks after (5 of 10 before), so two checks of the unchanged Mind still disagree; the cards now say what the agreement supports instead of nothing.
- **A looser card:** on 3 of 40 checks of the absent car allowance the model named the pay clause as covered, and the card shows "Your Documents support the employee's pay terms, not the parts about car allowance…" with the salary line, where it said "Nothing…".
- **Spend:** $0.09 on Haiku 5.5 and $2.16 on Sonnet 5.5 after; $1.78 for the before counts and their diagnosis.
