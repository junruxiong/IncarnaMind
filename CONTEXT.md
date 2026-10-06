# IncarnaMind

A notebook where people write Notes and ask Questions that are answered from their own Documents.

## Language

### Minds

**Mind**:
A notebook a User writes in: an ordered sequence of Blocks.
_Avoid_: session, chat

**Block**:
One unit of content in a Mind, positioned where the User placed it.
_Avoid_: message

**Note**:
A Block the User writes.
_Avoid_: text block

**Question**:
A Block in which the User asks for an Answer.
_Avoid_: query, prompt

**Answer**:
A Block generated in response to exactly one Question.
_Avoid_: output, response, reply

**Question context**:
What an Answer is generated from: every Block above its Question in the Mind, except Notes the User has excluded, plus Passages from the User's Documents.
_Avoid_: prompt, history

### Documents

**Document**:
A file the User has added for Answers to draw on.
_Avoid_: file, doc, source

**Passage**:
A short span of a Document that can be retrieved and cited.
_Avoid_: chunk, snippet

**Citation**:
A reference from an Answer to a Passage it drew on.
_Avoid_: source, reference

### People

**User**:
The person using IncarnaMind.
_Avoid_: account, UserAccount
