# Repeated checks of an unchanged Mind

`npm run eval:repeats` presses Check this Mind again and again on Minds nobody edits, with a real chat model and the app's real search, and counts what each sentence gets each time (#1608). A User who checks the same text twice expects the same verdicts, and never to be told a sentence a Document states wasn't found. Neither `npm test` nor CI runs it.

## What it does

1. A temporary data folder, as for the other evaluations, with the Library of a first run: the example's Documents (`createExamples`) beside the set's own agreement, so the search weighs the agreement against other Documents as it does for a newcomer. Documents are added with embeddings off and searched with the built-in reranking model, as Users have them by default. The interface is in the set's language.
2. For each set, a Mind with its sentences, one paragraph each.
3. Check this Mind, as the button does on an unchanged Mind: the first time the model names the claims and checks the new sentences ("changed"), every later time it checks them all ("all"). `INCARNAMIND_EVAL_REPEATS` times (10 by default).
4. After each check, every sentence's result is read from its mark, as the editor reads it, and named by how its card shows it:
   - **supported**, **contradicted**;
   - **partly**: unsupported, and the card says what the Documents support and what they don't, quoting the Passage the check read;
   - **differs**: unsupported, and the card quotes a line that says something different;
   - **not found**: unsupported, and the card says "I didn't find this in “{scope}”";
   - **not checked**, with why.
5. The report: per set and model, each sentence's outcomes and how many checks gave its most frequent verdict (partly, differs and not found are one verdict, unsupported), false Supported, not found on a sentence the Documents state, how many checks gave other verdicts than the check before, and every result that isn't the expected verdict in the check's own words. It goes into `eval/results/<start time>-repeats/` (`report.md`, `report.json`).

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

### 2026-10-23: the card says where it looked, and a part covered needs the same claim

Three versions on the same Documents, Library and search: `main` at bf47b4631 ("main"); the first partly-supported card for a sentence with no Citation ("first"); and this version ("now"), where the check's reply says in fields of its own whether a Passage is about the sentence's topic (`Topic: found` or `not found`) and whether the sentence names the Passage's party in general words (`Scope: same` or `wider`). A Supported reply with `Scope: wider` shows the partly-supported card; an Unsupported one with `Topic: not found` names no part covered, and its card says "I didn't find this in “{scope}”". Counts are of 10 checks; S supported, C contradicted, P the partly-supported card, N not found, D a line that says something different. `zh-control` didn't run on `main`.

| Set · sentence | Haiku main | Haiku first | Haiku now | Sonnet main | Sonnet first | Sonnet now |
|---|---|---|---|---|---|---|
| en · staff-home ("Staff…") | 10 S | 9 S, 1 N | 9 P, 1 S | 5 S, 5 N | 8 S, 2 P | 8 P, 2 S |
| en · car allowance (absent) | 10 N | 9 N, 1 P | 10 N | 10 N | 10 N | 10 N |
| en · holiday (wrong figure) | 10 C | 10 C | 10 C | 10 C | 10 C | 10 C |
| en · near miss (Mondays and Fridays) | 5 D, 5 C | 4 D, 6 C | 5 D, 5 C | 10 N | 10 P | 10 P |
| en-control · staff-home | not run | 10 S | 5 P, 4 S, 1 N | 8 S, 2 N | 9 S, 1 P | 10 P |
| en-control · employees-home | not run | 10 S | 9 P, 1 not checked | 10 S | 10 S | 10 P |
| zh · staff-home (员工…) | 9 S, 1 N | 9 S, 1 P | 9 P, 1 S | 9 S, 1 N | 10 S | 10 P |
| zh · notice (paraphrase) | 10 S | 10 S | 8 S, 2 P | 10 S | 10 S | 10 S |
| zh · car allowance (absent) | 10 N | 9 N, 1 P | 10 N | 10 N | 9 N, 1 P | 10 N |
| zh · holiday (wrong figure) | 10 C | 10 C | 10 C | 10 C | 10 C | 10 C |
| zh · near miss (每周一和周五) | 10 N | 10 P | 5 P, 5 C | 10 N | 10 P | 10 P |
| zh-control · staff-home | not run | 9 S, 1 P | 10 P | not run | 8 S, 2 P | 10 P |
| zh-control · employees-home | not run | 9 S, 1 D | 10 P | not run | 10 S | 10 P |

Every other sentence (the English notice, the agreement's own words, the controls' paraphrase and name) was Supported in every check of every version. No check gave a false Supported.

- **Not found, for a sentence the agreement states:** 9 checks on `main` (Sonnet 8, Haiku 1), 1 in the first version, 1 now (Haiku, English control).
- **Misses now, all on Haiku 5.5:** the English control's "Staff…" gets the partly card in 5 of 10 checks; the Chinese notice paraphrase gets a partly card in 2 of 10, where it was Supported in every check; the near miss is Contradicted in 5 of 10 in English, as on `main`, and in 5 of 10 in Chinese, where it was never Contradicted before.
- **Spend:** `main` $1.78 (with the diagnosis), the first version $2.26, now $2.28 (Haiku 5.5 $0.22, Sonnet 5.5 $2.06).
