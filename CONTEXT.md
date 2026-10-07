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
A file the User has added for Answers to draw on.
_Avoid_: file, doc, source

**Passage**:
A short span of a Document that can be retrieved and cited.
_Avoid_: chunk, snippet

**Citation**:
A reference, anchored in the text of an Answer, to the Passage a claim was drawn from: the page or two it cites, and a quote from the Passage that is checked against those pages. Copied into a Note, it stays a Citation; Question context shows it as a plain reference to its Document.
_Avoid_: source, reference

**Folder**:
A place where the User files Documents by hand. Folders can contain other Folders, and each Document is in at most one Folder.
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
A script that comes with a Skill, which an Answer can run on the User's computer. Each run asks the User first, unless they chose "always run" for that Skill; there is no sandbox, and a switch in Settings turns all Skill scripts off.
_Avoid_: tool script, code execution

**Built-in Skill**:
A Skill that ships with IncarnaMind and is updated with it: summarise a Document, a literature review across Documents, and a Mind to report. The User can turn one off, remove it and restore it, or duplicate it as their own, but not change it.
_Avoid_: default Skill, system Skill

### People

**User**:
The person using IncarnaMind.
_Avoid_: account, UserAccount
