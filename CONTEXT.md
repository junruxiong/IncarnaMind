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

### Changes

**Proposal**:
A change to a Mind, made by the AI or brought in from another person's Word file, that waits for the User to accept or reject it. Until it is accepted it does not count as part of the Mind, and it is out of date once the text it changes has changed.
_Avoid_: suggestion, edit, draft, diff

**Checkpoint**:
A saved copy of a Mind as it was just before an AI change, which the User can compare with another Checkpoint or restore from. Restoring one comes back as a Proposal, not as an overwrite.
_Avoid_: snapshot, version, backup, undo

### Checks

**Citation check**:
The test of whether a Citation's quote is found in the text at its Location. Its only failing result is quote not found.
_Avoid_: validation, verification

**Quote not found**:
The result of a Citation check when the Citation's quote is not in the text at its Location. It says nothing about whether the claim is true. "Not found" is used only for this.
_Avoid_: broken, invalid, failed

**Unsupported**:
The result for a sentence when Find support, Verify or note health finds nothing in the Documents that backs it up. It is not a finding that the sentence is wrong.
_Avoid_: not found, unverified, false

**Contradicted**:
The result for a sentence when Verify or contradiction watch finds a Passage that says the opposite. It is not the same as unsupported, where nothing is found either way.
_Avoid_: wrong, conflict, false

**Needs re-check**:
A check result whose sentence, evidence or scope has changed since it was made, so it is no longer known to hold.
_Avoid_: stale, outdated, expired

**Mark**:
The small sign beside a sentence that shows its check result: found, quote not found, unsupported or contradicted. In a Meeting, a Mark is also a flag the User sets at a moment in the Transcript.
_Avoid_: badge, flag, tick

### Tasks

**Run**:
One go of the Tool-calling loop, with its instructions, Tools and model. An Answer is one kind of Run; a Task is another.
_Avoid_: job, execution, session

**Task**:
A Run that goes on in the background, keeps its place if IncarnaMind is quit, and can be steered, paused or scheduled. Its results come back as Proposals or as new Documents.
_Avoid_: job, agent, process, background Answer

**Comparison table**:
A table in a Mind with one Document in each row and one Question in each column, whose cells are cited Answers. The command that builds one is Compare Documents.
_Avoid_: grid, spreadsheet

**Memory**:
What IncarnaMind has been asked to remember about the User, kept only when the User says so and open to read, change and delete.
_Avoid_: history, profile, learned preferences

### Meetings

**Meeting**:
A conversation the User records with the microphone and the computer's sound, which becomes a Transcript and the Notes written during it.
_Avoid_: call, recording, session

**Transcript**:
The words of a Meeting, in order and with times, kept as a Document so that a Citation can point at it. Its Location is a range of time.
_Avoid_: captions, subtitles, minutes

**Speaker**:
A person in a Transcript who is told apart from the others. A Speaker is named by the User, and a Citation to a Transcript names who said the quote.
_Avoid_: voice, participant, user

### Files

**Granted folder**:
A folder the User has allowed an Answer or a Task to read, once. It is not a Linked folder: its files do not become Documents unless the User keeps them.
_Avoid_: shared folder, workspace, permission

**Output location**:
The folder the User picks for the files IncarnaMind creates, such as exports and Documents a Task saves. IncarnaMind writes nowhere else without asking.
_Avoid_: export folder, downloads, save path

### People

**User**:
The person using IncarnaMind.
_Avoid_: account, UserAccount
