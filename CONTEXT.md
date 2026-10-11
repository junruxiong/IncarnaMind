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
What the User asks for an Answer: a Block in a Mind, or a message in a Conversation.
_Avoid_: query, prompt

**Answer**:
What is generated in response to exactly one Question, right after it: a Block in a Mind, or a message in a Conversation.
_Avoid_: output, response, reply

**Question context**:
What an Answer is generated from: in a Mind, every Block above its Question, except Notes the User has excluded; in a Conversation, its earlier Questions and Answers. Plus Passages from the Documents in the Question's Search scope.
_Avoid_: prompt, history

**Search scope**:
The Folders, Tags and individual Documents that a Question's Document search is limited to, resolved when the Question is asked: a Document filed into a Folder since is in it, one filed out is not. A Question in a Folder's Mind starts with the Folder as its Search scope, which the User can remove; a Question with no Search scope searches all Documents.
_Avoid_: filter, context

### The assistant

**Assistant**:
The one place in each window where the User asks Questions and reads their Answers, beside the Mind in focus and outside any tab; it reads that Mind and keeps its Conversations as its own history.
_Avoid_: chat panel, thread, copilot, bot

**Conversation**:
The assistant's history of one exchange: its messages, each a Question or an Answer, in order. It belongs to the Folder of the Mind its first Question was asked from, or to none; each Question records the Mind it was asked from. A Conversation is not a Mind: it is listed only in the assistant's History, never in the sidebar or a tab, and exists once it has a Question. New chat starts one.
_Avoid_: chat (except in "New chat"), thread, session

### Documents

**Document**:
A file on the User's computer, at its path, that IncarnaMind has indexed for Answers to draw on, or a Web page. A file stays where the User keeps it; IncarnaMind keeps the text it read from it. A Web page's snapshot is kept in IncarnaMind's data folder. The same file in two places is two Documents.
_Avoid_: file, doc, source

**Web page**:
A Document saved from a web address: a snapshot of the page's readable text, fetched from the User's computer when it is added, and again only when the User asks. Its Citations point at the snapshot, never at the live page, so they can always be checked.
_Avoid_: link, bookmark, clipping

**Missing Document**:
A Document whose file was removed while its folder is still there. New searches leave it out; its text, and the Citations that quote it, remain.
_Avoid_: deleted, broken

**Unavailable Document**:
A Document whose Linked folder can't be reached for now, such as on an unplugged drive. It is still searched, from its stored text; only its file can't be opened.
_Avoid_: offline, missing

**Linked folder**:
A folder on the User's computer that IncarnaMind keeps in sync: every supported file in it, at any depth, is a Document, and new, changed and removed files are picked up. IncarnaMind never changes or deletes a file in it, and adds a new file there only once the User allows it for that folder. Files a cloud drive keeps online only are indexed when the User asks.
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

**Footnote**:
A note the User writes at a point in the text, shown there by its number; its text can hold Citations. Footnotes and Citations share one numbering, in the order they appear, in the Mind and in every export, where a Citation's footnote is its reference.
_Avoid_: endnote, annotation, comment

**Reference record**:
What a citation style needs to cite a Document: its type, title, authors, date, where it was published, volume, issue, pages, publisher, DOI, URL and language. It is read from the Document's file, the User can edit every field (an edit is never replaced by a later read), and Look up can fill its empty fields from the Document's DOI or arXiv ID. A record of a known type that lacks its title, authors or date "needs details". A file exported from Zotero, EndNote or Mendeley (CSL JSON, BibTeX or RIS) can be imported: each reference is matched to a Document, or kept as a record with no Document until its PDF is added; the User's edits win over an import.
_Avoid_: metadata, bibliographic entry, citation (a Citation is the reference in an Answer)

**Citation style**:
How a Mind's Citations read, in its text and its exports, and how its Bibliography lists their Documents: a style of the Citation Style Language, such as APA 7, IEEE or GB/T 7714—2015, written from the Documents' reference records, or Numbered, IncarnaMind's own (numbered markers, each a footnote naming its Document and Location). Each Mind has one; one that hasn't chosen is Numbered, and a new Mind starts in the style last chosen for a Mind in its Folder. A style changes how a Citation reads, never its check.
_Avoid_: citation format, reference style

**Bibliography**:
Every Document a Mind cites, once each, in its citation style: the References at the Mind's end, kept up to date, and the end of its exports. A Mind in Numbered shows none. A Document whose reference record is too thin to cite is listed by its name, marked "needs details".
_Avoid_: works cited (except as a style's own heading), sources list

**Source provider**:
A public service that Find sources searches for works to add to the Library: OpenAlex, arXiv and PubMed (Crossref looks a DOI up); openFDA and DailyMed for drug labels and ClinicalTrials.gov for trials, which are asked only when the User chooses them; later, services for laws, regulations and filings. A work with an open-access copy is added as a Document, which keeps the provider it came from and the date of the version it holds; any other work, as its Reference record alone. Each Source provider is asked once for consent before anything is sent to it.
_Avoid_: source (a Document is not one), database, provider on its own (an AI model's provider)

**Location**:
Where in a Document a Citation points, in the unit the Document's own readers use: one or two pages of a PDF, one or two slides of a deck (speaker notes count as part of their slide), a sheet and a range of rows of a spreadsheet, the section a quote sits under in a Word file or Markdown (for a quote of a Word comment, the section the comment is anchored in, and that it is a comment by its author), the section and paragraphs of a Web page, or a range of lines in plain text.
_Avoid_: page (for anything that isn't a PDF), position, anchor

**Folder**:
A project inside IncarnaMind: a named place for Documents and Minds, with a description of what belongs there. Each Document and each Mind belongs to one Folder or is Not in a Folder; this does not change where a Document's original file is stored. A Mind in a Folder searches the Folder by default; deleting a Folder moves its Documents and Minds to Not in a Folder.
_Avoid_: project (in the interface), group (a Folder's former name; a Group is now a column of tabs), collection, category

**Not in a Folder**:
The one place for every Document and Mind in no Folder. A Mind there searches all Documents.
_Avoid_: Unsorted, Uncategorised, Inbox

**Suggested Folder**:
A Folder that Organize proposes for a Document, which waits for the User to accept it (it becomes the User's choice) or reject it (the Document stays Not in a Folder and that Folder isn't suggested for it again). While it waits, the Document counts as in that Folder, for listings and for the Folder's Search scope, marked as a suggestion.
_Avoid_: automatic Folder, proposed Folder

**Source location**:
The original file or Linked folder from which IncarnaMind reads Documents. Its folders mirror the disk and are independent of the Folders used to organise Documents inside IncarnaMind.

**Tag**:
A label with a short description that a Document can carry. A Document can have many Tags. IncarnaMind applies Tags automatically, and the User can add or remove them.
_Avoid_: category, label, class

**Topic**:
The earlier design for a generated two-level subject tree. The current Library uses flat Folders instead (direction revised 2026-10-08), and the dormant Topic code was removed on 2026-10-10 (ADR-0012).
_Avoid_: cluster, category, collection, group

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

**Outside agent**:
An AI agent the User runs themselves, such as Claude Code or Codex, connected to IncarnaMind so it can search and read the Folders the User shares with it and cite them with Citations IncarnaMind checks. The User connects each one in Settings, with its own Folders (none until chosen), its own consent and a list of every call it made; disconnecting it stops it at once, and Local mode pauses it.
_Avoid_: harness, plugin, client, bot

**Built-in Skill**:
A Skill that ships with IncarnaMind and is updated with it: summarise a Document, a literature review across Documents, and a Mind to report. The User can turn one off, remove it and restore it, or duplicate it as their own, but not change it.
_Avoid_: default Skill, system Skill

### Changes

**Proposal**:
A change to a Mind, made by the AI or brought in from another person's Word file, that waits for the User to accept or reject it. Until it is accepted it does not count as part of the Mind, and it is out of date once the text it changes has changed. Each Proposal is one change in one place (the words it removes and those it adds there, or a whole Block), accepted or rejected on its own; one request, such as an Edit with AI, can make several. The interface calls them changes: "3 changes from the assistant".
_Avoid_: suggestion, edit, draft, diff

**Checkpoint**:
A saved copy of a Mind as it was just before an AI change, which the User can compare with another Checkpoint or restore from. Restoring one comes back as a Proposal, not as an overwrite.
_Avoid_: snapshot, version, backup, undo

**Recently deleted**:
Where a deleted Mind, Conversation, Document, Folder or Tag waits for 30 days, from where it can be restored or deleted permanently. Deleting never touches the User's files.
_Avoid_: trash, bin, archive

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

### Privacy

**Usage data**:
Anonymous events about using IncarnaMind, such as "a Question was asked with a local model", sent only while the User agrees, under a random install ID. It never holds a Document's content or name, a Question, an Answer or anything the User typed.
_Avoid_: telemetry, tracking

**AI use record**:
What the AI put into a Mind, or proposed and the User turned down, and what became of it (kept, edited, deleted), kept per Mind on this computer. It holds kinds, sizes, the model and times, never text, and is never sent, whatever the User chose about Usage data.
_Avoid_: AI log, AI history

### Window

**Tab**:
What a Mind, the Library or a Document is open in. A preview tab, its title in italics, is replaced by the next Document opened in its group until the User keeps it; a pinned tab stays at the left of its group.
_Avoid_: pane, page, view

**Group**:
One of up to three columns of tabs side by side in a window, each with its own tab strip; in Chinese, 栏. A narrow window shows its groups merged into one.
_Avoid_: pane, split, column

### People

**User**:
The person using IncarnaMind.
_Avoid_: account, UserAccount
