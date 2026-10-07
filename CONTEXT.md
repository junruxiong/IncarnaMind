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
What an Answer is generated from: every Block above its Question in the Mind, except Notes the User has excluded, plus Passages from the Documents in the Question's Search scope.
_Avoid_: prompt, history

**Search scope**:
The Folders, Tags and individual Documents that a Question's Document search is limited to. A Question with no Search scope searches all Documents.
_Avoid_: filter, context

### Documents

**Document**:
A file on the User's computer, at its path, that IncarnaMind has indexed for Answers to draw on. The file stays where the User keeps it; IncarnaMind keeps the text it read from it. The same file in two places is two Documents.
_Avoid_: file, doc, source

**Missing Document**:
A Document whose file was removed while its folder is still there. New searches leave it out; its text, and the Citations that quote it, remain.
_Avoid_: deleted, broken

**Unavailable Document**:
A Document whose Linked folder can't be reached for now, such as on an unplugged drive. It is still searched, from its stored text; only its file can't be opened.
_Avoid_: offline, missing

**Linked folder**:
A folder on the User's computer that IncarnaMind keeps in sync: every supported file in it, at any depth, is a Document, and new, changed and removed files are picked up. IncarnaMind never changes anything in it. Files a cloud drive keeps online only are indexed when the User asks.
_Avoid_: workspace, vault, library, source

**Other Documents**:
The Documents added on their own, outside any Linked folder.
_Avoid_: loose files, imports

**Passage**:
A short span of a Document that can be retrieved and cited.
_Avoid_: chunk, snippet

**Citation**:
A reference, anchored in the text of an Answer, to the Passage a claim was drawn from: the page or two it cites, and a quote from the Passage that is checked against those pages. Copied into a Note, it stays a Citation; Question context shows it as a plain reference to its Document.
_Avoid_: source, reference

**Folder**:
A folder inside a Linked folder, as it is on disk. A Document is in the Folder its file is in; a file added on its own is in no Folder.
_Avoid_: collection, directory, category

**Tag**:
A label with a short description that a Document can carry. A Document can have many Tags. IncarnaMind applies Tags automatically, and the User can add or remove them.
_Avoid_: category, label, class

### Tools

**Tool**:
A single action an Answer can call while it is being generated, provided by Document search, a Connector or a Skill.
_Avoid_: function, action

**Connector**:
An external service the User has connected, through which Answers can look things up or make changes.
_Avoid_: MCP server, plugin, integration, connection

**Skill**:
A packaged description of how to do a particular task, optionally with reference files and scripts, that an Answer can follow.
_Avoid_: plugin, command

**Skill script**:
A script that comes with a Skill, which an Answer can run on the User's computer with the User's approval.
_Avoid_: tool script, code execution

**Built-in Skill**:
A Skill that ships with IncarnaMind and is updated with it: summarise a Document, a literature review across Documents, and a Mind to report. The User can turn one off, remove it and restore it, or duplicate it as their own, but not change it.
_Avoid_: default Skill, system Skill

### People

**User**:
The person using IncarnaMind.
_Avoid_: account, UserAccount
