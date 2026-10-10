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
_Avoid_: workspace, vault, source

**Other Documents**:
The Documents added on their own, outside any Linked folder.
_Avoid_: loose files, imports

**Passage**:
A short span of a Document that can be retrieved and cited.
_Avoid_: chunk, snippet

**Citation**:
A reference, anchored in the text of an Answer, to the Passage a claim was drawn from: the Location it cites, and a quote from the Passage that is checked against the text at that Location. Copied into a Note, it stays a Citation; Question context shows it as a plain reference to its Document.
_Avoid_: source, reference

**Location**:
Where in a Document a Citation points, in the unit the Document's own readers use: one or two pages of a PDF, one or two slides of a deck (speaker notes count as part of their slide), a sheet and a range of rows of a spreadsheet, the section a quote sits under in a Word file or Markdown, or a range of lines in plain text.
_Avoid_: page (for anything that isn't a PDF), position, anchor

**Folder**:
A named place for Documents inside IncarnaMind, with a description of what belongs there. Each Document belongs to one Folder or is Unsorted; this does not change where its original file is stored.
_Avoid_: group, collection, category

**Source location**:
The original file or Linked folder from which IncarnaMind reads Documents. Its folders mirror the disk and are independent of the Folders used to organise Documents inside IncarnaMind.

**Tag**:
A label with a short description that a Document can carry. A Document can have many Tags. IncarnaMind applies Tags automatically, and the User can add or remove them.
_Avoid_: category, label, class

**Topic**:
The earlier design for a generated two-level subject tree. The current Library uses flat Folders instead (direction revised 2026-10-08), and the dormant Topic code was removed on 2026-10-10 (ADR-0012).
_Avoid_: cluster, category, collection, group

**Group**:
The former name for an in-app Folder. Users now see Folders throughout document organisation.

**Library**:
The view of all of a User's Documents, organised into Folders with visible Tags. Users can search names or Tags, manage Folders, and organise Documents automatically or correct them by hand; manual choices are kept.
_Avoid_: archive, index, vault

### Tools

**Tool**:
A single action an Answer can call while it is being generated, provided by Document search, a Connector or a Skill.
_Avoid_: function, action

**Effect**:
What a Tool call can do beyond the conversation it is part of: read data, write (change something), run code on a computer, or send data over the network, and where (some of the User's folders, a web host, a Connector's service, the Documents, a Mind). Whether a call asks the User first depends on its Effects.
_Avoid_: permission, capability, side effect

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
