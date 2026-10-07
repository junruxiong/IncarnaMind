---
name: summarise-document
description: Writes a structured summary of one Document, or of each Document in the Question's Search scope, covering its key claims (each with a Citation), its methods or approach, its limitations and the questions it leaves open. Use when the User asks to summarise, digest or give an overview of a paper, report or other Document.
license: Apache-2.0
metadata:
  author: IncarnaMind
---

# Summarise a Document

Write a structured summary of a Document the User added, drawn only from what its Passages say.

## Which Documents

- If the Question names a Document, or its Search scope holds a single Document, summarise that one.
- If the Search scope holds several Documents and the Question doesn't pick one, summarise each in turn, in the same structure but shorter, then add a short paragraph on how they relate.
- If you can't tell which Document is meant, summarise the one the Passages point to and say which one you chose.

## Gather the material first

Search the Documents before writing anything (search_documents searches only the Documents in the Question's Search scope):

- Make several focused searches rather than one broad one: the Document's main subject, its aims or research question, its methods or data, its findings or results, its limitations, and its conclusions or future work. Run several at once where you can.
- After the first results, search again with the Document's own terms, and with its name if several Documents are in scope.
- If you were given Passages instead of a search Tool, work from those.

## Write the summary

Write it in the language the Question is written in, even when the Document is in another language. Translate the headings below into that language too.

Start with a heading naming the Document, then one or two sentences on what it is (a paper, a report, a book chapter, …), its subject, and its main contribution. Then these sections:

1. **Key claims**: three to seven bullet points, each one claim or finding of the Document, in your own words, each with a Citation. Keep the numbers, conditions and qualifications the Document gives.
2. **Methods or approach**: how the Document reaches its claims: its data, experiments, sources, analysis or line of argument.
3. **Limitations**: the limitations the Document states itself, with Citations. Limitations you notice yourself go after them, clearly labelled as your own assessment and without a Citation.
4. **Open questions**: what the Document leaves open or names as future work, with Citations, then any questions you see, labelled as yours.

Keep it short: a summary, not a rewrite. Bullet points are fine; don't use tables.

## Citations and honesty

- Cite every statement you take from a Document, the way your instructions for citing say: markers that point to the Passages you found. Don't write a list of sources or a bibliography, and never cite a Passage for something it doesn't say.
- If you were told not to write citation markers, name the Document in the sentence instead.
- Don't add details the Passages don't give: no names, numbers, dates or sample sizes from memory.
- When the Passages found don't cover a section, say so plainly in that section (for example: "The Passages found don't describe the methods.") instead of guessing. If the Documents don't support what the Question assumes, say that too.
- Report the Document's views as the Document's. Keep your own views out, apart from the labelled assessments above.
