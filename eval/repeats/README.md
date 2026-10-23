# Repeated checks of an unchanged Mind

`npm run eval:repeats` presses Check this Mind again and again on Minds nobody edits, with a real chat model and the app's real search, and counts what each sentence gets each time (#1608). A User who checks the same text twice expects the same verdicts, and never to be told a sentence a Document states wasn't found. Neither `npm test` nor CI runs it.

## What it does

1. A temporary data folder, as for the other evaluations, with the Library of a first run: the example's Documents (`createExamples`) beside the set's own agreement (and, for the conflict Minds, a short flat-and-laptop agreement between the same parties, which only their Library holds), so the search weighs the agreement against other Documents as it does for a newcomer. Documents are added with embeddings off and searched with the built-in reranking model, as Users have them by default. The interface is in the set's language.
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
| `en-conflicts`, `zh-conflicts` | Genuine contradictions of the agreements, one of each kind, each expected Contradicted: a number, a date and a year, a negation, a unit (per month for per year), the parties reversed (who reports to whom, who pays whom, employer and employee, seller and buyer, landlord and tenant), a name, a condition (seven days' notice for 24 hours), a scope word ("only during employment"), a place (the governing law), an entitlement turned off, and "unlimited"; in English also the pension shares swapped and a duty for a permission ("must" for "may"), in Chinese who pays the social insurance swapped |

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

**The conflicts:** a change to a verdict's instructions or to a judgement is measured against the conflicts it loses, not only the near misses it fixes. The report gives "Genuine conflicts Contradicted: n of m", and counts a conflict Supported as a false Supported. A sentence the check misses today is marked with its issue (`knownMiss` in `cases.ts`), stays in its set, and is listed with its counts under a heading of its own, "Known misses (#…)", outside every other count; take the mark off once the issue is fixed. Marked today, as #1668: "must" for "may"; employer and employee reversed (English); who pays the social insurance and who pays the rent reversed, landlord and tenant reversed, and the negation the claims leave out (Chinese).

## Scope of a check from a Mind

`npm run eval:scope` checks a Mind that summarises one Document, in a Library of that Document and 5 or 20 other invented Documents, to see whether searching every Document, as a Mind in no Folder does, costs it verdicts. The cases are `unfiled-5`, `filed-5`, `unfiled-20` and `filed-20`: the Mind either in no Folder or filed into the Folder that holds the agreement, with the other Documents outside it. `scoped-5` and `scoped-20` are the Mind in no Folder with the agreement as the Search scope the User set for it in the composer (#1623). The same six with the Chinese 劳动合同 and Documents (`scopeDocumentsZh.ts`) have the prefix `zh-`. `unfiled-no-agreements-17` is the 17 other Documents that remain once the other agreements are left out.

- The Documents are the English agreement of `cases.ts` and the first 5 or all 20 of `scopeDocuments.ts`: two or three more employment agreements in the agreement's layout with other figures, policies an employer keeps, and Documents of other kinds. None states what a sentence says (`tests/eval/scopeScoring.test.ts`). There are no example Documents, so the number of others is exact.
- The sentences are four of the `en` set, worded as there: the notice paraphrase and the agreement's own words, which it states, and the car allowance it doesn't mention and the holiday sentence with a wrong figure, which it doesn't state.
- The report counts, per case and model, the faithful sentences that didn't come back Supported, the other sentences that did, and for every wrong check whether a Passage handed to the model held the clause that settles the sentence. The model's requests are watched for the Passages (`passagesHanded.ts`), and the Passages of each Library are kept in `passages-<case>.json`.
- `INCARNAMIND_EVAL_SCOPE_CASES` picks cases by id, `INCARNAMIND_EVAL_SCOPE_BUDGET` stops the run past that many dollars (1), and `INCARNAMIND_EVAL_REPEATS` is the checks a case (10). Ten checks of one case cost $0.02 to $0.05 on Claude Haiku 5.5 and $0.30 to $0.60 on Claude Sonnet 5.5, a Mind in no Folder the most.

## Results

### 2026-10-23: a Mind's chosen scope narrows its check (#1623)

The scope cases on this version, where a check from a Mind searches the Search scope the User set for it, else its Folder, else every Document. "Before" is the measurement on the version this one is built on, where a Mind in no Folder searched every Document whatever scope it had; "filed" is the same Mind in the agreement's Folder, which this version doesn't change. Counts are of 20 checks of the two faithful sentences (false Unsupported) and of the two the agreement doesn't state (false Supported); 10 checks a case.

| Model | Language | Others | Before: in no Folder | Now: in no Folder, scope the agreement | Now: filed | False Supported now |
|---|---|---|---|---|---|---|
| Haiku 5.5 | English | 5 | 9 of 20 | 0 of 20 | 0 of 20 | 0 of 40 |
| Haiku 5.5 | English | 20 | 13 of 20 | 0 of 20 | 0 of 20 | 0 of 40 |
| Haiku 5.5 | Chinese | 5 | 10 of 20 | 1 of 20 | 0 of 20 | 0 of 40 |
| Haiku 5.5 | Chinese | 20 | 11 of 20 | 0 of 20 | 0 of 20 | 0 of 40 |
| Sonnet 5.5 | English | 20 | 0 of 20 | 0 of 20 | not run | 0 of 20 |
| Sonnet 5.5 | Chinese | 20 | 1 of 20 | 0 of 20 | not run | 0 of 20 |

- Every check of a Mind with its scope set was handed the agreement alone, as every filed check was. The car allowance was "I didn’t find this" and the holiday sentence Contradicted in all 140 checks.
- The one miss is Haiku's partly-supported card on the Chinese notice paraphrase, with the agreement alone in hand: it took the sentence as wider than the agreement's two parties ("Not covered: 适用于所有劳动者").
- A first run on this change before the check's newer reply wording was merged in gave 0 of 20 in every case, the Chinese one with 5 others included.
- Spend at the app's own count: $1.57 for both runs (Haiku $0.27, Sonnet $1.30).

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
- **With only the strict "neither" downgrading** (see the conflicts below), Haiku 5.5, 20 checks of `en` and `zh`, 10 of each control: the near miss Contradicted in 2 of 20 in Chinese and 0 of 20 in English; the own words and the notice paraphrase Supported in 20 of 20 in both languages; "Staff…" partly in 19 of 20 and 20 of 20, and in 10 of 10 in both control Minds, "Employees…" too; the holiday Contradicted in 20 of 20 and 19 of 20 (one "D"); the car allowance "I didn't find this" in all 40; no false Supported. Sonnet 5.5, 5 checks of the four sets: every sentence as expected in every check.
- **Genuine conflicts** (`en-conflicts`, `zh-conflicts`, added after): with the strict judgement downgrading on any word but "conflict", the Chinese reversed reporting line, "人力资源总监向乙方汇报。", got the partly-supported card in 5 of 5 checks: writing it in English, the strict judgement read the direction backwards and said "same" (8 of 10 requests alone). So only its "neither" (the quote doesn't state the detail) downgrades a contradiction; its "same" leaves it Contradicted. Since then, on Haiku 5.5, 5 checks: Contradicted in 80 of 80 in English and 63 of 65 in Chinese, known misses left out, the reversed reporting line in 5 of 5. Asked alone 10 times each, the strict judgement said "neither" to none of the reversed parties. What isn't Contradicted comes before it: a reversed party Supported by the first verdict (in English employer and employee once in 5; in Chinese who pays the rent twice and landlord and tenant once), employer and employee reversed in Chinese "I didn't find this" twice (the first confirmation said "neither"), and the Chinese negation left unmarked in every check (the claims weren't named with it). One Chinese check of the date was not checked in an earlier run: its reply wrote the verdict lines beside `cite` with no marker, which the Answer engine drops as a preamble, so what was left had no verdict. The two known misses (#1668), "must" for "may" and the swapped social insurance, were Supported in 5 of 5 and 4 of 5, by the first verdict. Sonnet 5.5, one check of each: 16 of 16 in English and 13 of 13 in Chinese, known misses left out (of them, employer and employee reversed got the partly-supported card from the check itself, the strict judgement alone saying "conflict" to it in 5 of 5; the others were Contradicted).
- **Why a second judgement and not the existing one:** asked for one word, both existing judgements called the near miss a conflict in 10 of 10 requests each, in English and Chinese, on Haiku 5.5 with thinking off, as checks run. The strict one did in 0 of 10 in English and 1 of 30 in Chinese; with its lines in the sentence's language, 9 of 60 in Chinese. It still called every genuine conflict tried a conflict, 10 of 10 each: the two holiday figures, three days a week, "Mondays, Wednesdays and Fridays" against two days a week (in English and Chinese), other hours, another manager, a later date, another pension share and a longer probation; and Sonnet 5.5 judged all 12 cases tried as expected. Asking it for the parts in the same request swayed its word (the Chinese near miss a conflict in 15 of 20), so the parts are a request of their own.
- **Time:** the strict judgement is one more request for a sentence the check calls contradicted: a median of 1.0 to 1.2 s (90th percentile 1.5 to 2.0 s) on Haiku 5.5, 1.9 s (3.7 s) on Sonnet 5.5. The parts request, when needed, about 1 s. Sentences are checked several at once, so a Mind's check takes longer only by what its slowest sentence adds.
- **Spend:** $6.22 in all: Haiku 5.5 $1.52 (the runs while building it, the conflict sets and the diagnosis), the judgement trials $0.65 (Sonnet 5.5 $0.42 of it), Sonnet 5.5's runs of the sets $4.05.

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
