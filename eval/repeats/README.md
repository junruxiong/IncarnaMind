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
   - **absent**: unsupported, and the card says "I didn’t find this in “{scope}”";
   - **not checked**, with why.
5. The report: per set and model, each sentence's outcomes and how many checks gave its most frequent verdict (partly, differs and absent are one verdict, unsupported), false Supported, absent on a sentence the Documents state, the partly-supported card on a sentence whose topic the Documents don't hold, how many checks gave other verdicts than the check before, and every result that isn't the expected verdict in the check's own words. It goes into `eval/results/<start time>-repeats/` (`report.md`, `report.json`).

The cases are in `cases.ts`; `repeat.ts` checks any Mind's sentences in any core, so other sets can use it with their own Documents.

## The sets

An employment agreement in English and a 劳动合同 in Chinese (two-page PDFs, invented for this evaluation), and six Minds:

| Set | Sentences |
|---|---|
| `en`, `zh` | The three sentences of #1608 (notice as a paraphrase; holiday with a wrong figure; "Staff" for "the Employee"), the agreement's own words, a car allowance it doesn't mention, and a near miss that names days it doesn't |
| `en-control`, `zh-control` | The home-working clause four ways: a paraphrase about the same Employee, the Employee by name, "Staff" (员工) and "Employees" (职工) |
| `en-conflicts`, `zh-conflicts` | Genuine contradictions of the agreement, one of each kind, each expected Contradicted: a number, a date and a year, a negation, a unit (per month for per year), a party (who reports to whom), a name, a condition (seven days' notice for 24 hours), a scope word ("only during employment"), a place (the governing law), the pension shares swapped, an entitlement turned off, and "unlimited"; in English also a duty for a permission ("must" for "may"), in Chinese the two parties swapped |

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

**Cost and time:** ten checks of the first four Minds take about 5 minutes and $0.10 on Claude Haiku 5.5, and about 9 minutes and $2.20 on Claude Sonnet 5.5. The two conflict Minds add about 9 minutes and $0.25 on Haiku 5.5, and roughly $4 on Sonnet 5.5 (from the other sets' ratio, not measured), so run them there with fewer checks.

**The conflicts:** a change to a verdict's instructions or to a judgement is measured against the conflicts it loses, not only the near misses it fixes. The report gives "Genuine conflicts Contradicted: n of m", and counts a conflict Supported as a false Supported. A sentence the check misses today is marked with its issue (`knownMiss` in `cases.ts`) and reported on a line of its own, "Known miss (#…), not counted", outside every other count; take the mark off once the issue is fixed.

## Results

### 2026-10-23: a contradiction needs a second, strict judgement (#1647)

On the same Documents, Library and search as below. "Before" is the version below ("now" there: Haiku 5.5's two runs, Sonnet 5.5's one); "after" is this version, whose changes are:

- A sentence the check calls contradicted, and the first confirmation upholds, is judged once more beside its quote, by a judgement that writes the sentence's detail ("on Mondays and Fridays (two named days)") and what the quote states of it before its word, in English. Only "conflict" keeps it Contradicted. Otherwise the two judgements disagree and the card is the partly-supported one, quoting the Passage, with the reply's own parts or, when it named none, parts asked for once. A judgement that fails or can't be read leaves the sentence not checked, with why.
- The reply's `Scope` field names the sentence's own word for the party before `same` or `wider` ("Scope: Staff, wider"; "Scope: none, same" when the sentence names none).
- A Supported reply with a wider scope whose parts can't be shown asks for them once.

Haiku 5.5 ran twice after (runs 1 and 2, the last two commits; the second only adds the third change). Sonnet 5.5 ran once after; `en-control` again on the last commit, the run shown. Counts are of 10 checks, as below.

| Set · sentence | Haiku before (1; 2) | Haiku after (1; 2) | Sonnet before | Sonnet after |
|---|---|---|---|---|
| en · own words | 8 S, 2 P; 10 S | 10 S; 10 S | 10 S | 10 S |
| zh · notice (paraphrase) | 10 S; 10 S | 10 S; 10 S | 10 S | 10 S |
| zh · near miss (每周一和周五) | 2 P, 8 C; 3 P, 7 C | 10 P; 10 P | 10 P | 10 P |
| en · near miss (Mondays and Fridays) | 5 D, 5 C; 7 D, 3 C | 5 P, 5 D; 4 P, 5 D, 1 C | 10 P | 10 P |
| en · staff-home ("Staff…") | 10 P; 10 P | 10 P; 10 P | 10 P | 10 P |
| en-control · staff-home | 10 P; 10 P | 10 P; 9 P, 1 S | 10 P | 10 P |
| en-control · employees-home | 9 P, 1 S; 9 P, 1 not checked | 10 P; 9 P, 1 S | 10 P | 10 P |
| zh · staff-home (员工…) | 10 P; 10 P | 10 P; 10 P | 10 P | 10 P |
| zh-control · staff-home | 10 P; 10 P | 10 P; 10 P | 10 P | 10 P |
| zh-control · employees-home | 10 P; 10 P | 10 P; 9 P, 1 D | 10 P | 10 P |
| en · holiday (wrong figure) | 10 C; 10 C | 10 C; 10 C | 10 C | 10 C |
| zh · holiday (wrong figure) | 10 C; 10 C | 10 C; 10 C | 10 C | 10 C |
| en · car allowance (absent) | 10 A; 10 A | 10 A; 10 A | 10 A | 10 A |
| zh · car allowance (absent) | 10 A; 10 A | 10 A; 10 A | 10 A | 10 A |

Every other sentence was Supported in every check of every run. No check gave a false Supported, and none after was not checked.

- **The near miss:** rarely Contradicted on Haiku 5.5 after, not never: 3 of 30 checks in Chinese and 1 of 30 in English in a re-test (0 of 20 and 1 of 20 in the runs above), against 15 of 20 and 8 of 20 before. Its "D" cards are unsupported verdicts whose nearest line the existing nearest-line judgement (#770) calls different, as before.
- **Genuine conflicts** (`en-conflicts`, `zh-conflicts`, 5 checks on Haiku 5.5, added after): Contradicted in 64 of 65 checks in English and 54 of 60 in Chinese, with no false Supported. The Chinese reversed reporting line, "人力资源总监向乙方汇报。", gets the partly-supported card in 5 of 5: the check and the first confirmation call it a conflict, and the strict judgement, writing it in English, reads the direction backwards and says "same". The English one gets that card once in 5. One Chinese check of the date was not checked: its reply wrote the verdict lines beside `cite` with no marker, which the Answer engine drops as a preamble, so what was left had no verdict. The two known misses (#1668), "must" for "may" and the swapped parties, were Supported in 4 of 5 and 5 of 5, by the first verdict, before any judgement.
- **Why a second judgement and not the existing one:** asked for one word, both existing judgements called the near miss a conflict in 10 of 10 requests each, in English and Chinese, on Haiku 5.5 with thinking off, as checks run. The strict one did in 0 of 10 in English and 1 of 30 in Chinese; with its lines in the sentence's language, 9 of 60 in Chinese. It still called every genuine conflict tried a conflict, 10 of 10 each: the two holiday figures, three days a week, "Mondays, Wednesdays and Fridays" against two days a week (in English and Chinese), other hours, another manager, a later date, another pension share and a longer probation; and Sonnet 5.5 judged all 12 cases tried as expected. Asking it for the parts in the same request swayed its word (the Chinese near miss a conflict in 15 of 20), so the parts are a request of their own.
- **Time:** the strict judgement is one more request for a sentence the check calls contradicted: a median of 1.0 to 1.2 s (90th percentile 1.5 to 2.0 s) on Haiku 5.5, 1.9 s (3.7 s) on Sonnet 5.5. The parts request, when needed, about 1 s. Sentences are checked several at once, so a Mind's check takes longer only by what its slowest sentence adds.
- **Spend:** $3.27 in all: Haiku 5.5 $0.52 (four runs of the four sets while building it, and the diagnosis), the judgement trials $0.32 (Sonnet 5.5 $0.09 of it), Sonnet 5.5 $2.43 (the four sets, then `en-control` again).

### 2026-10-23: the card says where it looked, and a part covered needs the same claim

Three versions on the same Documents, Library and search: `main` at bf47b4631 ("main"); the first partly-supported card for a sentence with no Citation ("first"); and this version ("now"). Now, the check's reply says in fields of its own whether a Passage is about the sentence's topic (`Topic: found` or `not found`) and whether the sentence names the Passage's party in general words (`Scope: same` or `wider`); "Covered" and "Not covered" are asked for only when a Passage states part of what the sentence itself says, never for a neighbouring thing. A Supported reply with `Scope: wider` shows the partly-supported card; a sentence no Passage states says "I didn’t find this in “{scope}”". Haiku 5.5 ran twice now (run 1; run 2). Counts are of 10 checks; S supported, C contradicted, P the partly-supported card, A absent, D a line that says something different. `zh-control` didn't run on `main`.

| Set · sentence | Haiku main | Haiku first | Haiku now (1; 2) | Sonnet main | Sonnet first | Sonnet now |
|---|---|---|---|---|---|---|
| en · staff-home ("Staff…") | 10 S | 9 S, 1 A | 10 P; 10 P | 5 S, 5 A | 8 S, 2 P | 10 P |
| en · own words | 10 S | 10 S | 8 S, 2 P; 10 S | 10 S | 10 S | 10 S |
| en · car allowance (absent) | 10 A | 9 A, 1 P | 10 A; 10 A | 10 A | 10 A | 10 A |
| en · holiday (wrong figure) | 10 C | 10 C | 10 C; 10 C | 10 C | 10 C | 10 C |
| en · near miss (Mondays and Fridays) | 5 D, 5 C | 4 D, 6 C | 5 D, 5 C; 7 D, 3 C | 10 A | 10 P | 10 P |
| en-control · staff-home | not run | 10 S | 10 P; 10 P | 8 S, 2 A | 9 S, 1 P | 10 P |
| en-control · employees-home | not run | 10 S | 9 P, 1 S; 9 P, 1 not checked | 10 S | 10 S | 10 P |
| zh · staff-home (员工…) | 9 S, 1 A | 9 S, 1 P | 10 P; 10 P | 9 S, 1 A | 10 S | 10 P |
| zh · notice (paraphrase) | 10 S | 10 S | 10 S; 10 S | 10 S | 10 S | 10 S |
| zh · car allowance (absent) | 10 A | 9 A, 1 P | 10 A; 10 A | 10 A | 9 A, 1 P | 10 A |
| zh · holiday (wrong figure) | 10 C | 10 C | 10 C; 10 C | 10 C | 10 C | 10 C |
| zh · near miss (每周一和周五) | 10 A | 10 P | 2 P, 8 C; 3 P, 7 C | 10 A | 10 P | 10 P |
| zh-control · staff-home | not run | 9 S, 1 P | 10 P; 10 P | not run | 8 S, 2 P | 10 P |
| zh-control · employees-home | not run | 9 S, 1 D | 10 P; 10 P | not run | 10 S | 10 P |

Every other sentence (the notice in English, the controls' paraphrase and name) was Supported in every check of every version. No check gave a false Supported.

- **The partly-supported card on the absent car allowance:** 0 of 40 checks on Haiku 5.5 and 0 of 20 on Sonnet 5.5 now, against 2 of 20 and 1 of 20 in the first version.
- **Absent, for a sentence the agreement states:** 9 checks on `main` (Sonnet 8, Haiku 1), 1 in the first version, 0 now.
- **Misses now, all on Haiku 5.5:** the near miss is Contradicted in 8 and 7 of 10 in Chinese (never on `main`, 5 of 10 in the version pushed before this one) and in 5 and 3 of 10 in English (5 on `main`); the agreement's own words get a partly card in 2 of 10 in one English run.
- **Spend:** `main` $1.78 (with the diagnosis), the first version $2.26, a version in between $2.28, now $2.19 (Haiku 5.5 $0.23, Sonnet 5.5 $1.96).
