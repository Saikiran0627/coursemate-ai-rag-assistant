# Lecture 5: Responsible AI and Security

Page 1

CourseMate should protect privacy by avoiding unnecessary collection of student
data. Uploaded course documents should be treated as user-provided content. API
keys, passwords, and private student information should not be committed to the
repository or displayed in the interface.

Page 2

Prompt injection is a risk for RAG systems because retrieved documents are
untrusted input. The system prompt should remain separate from document text.
The assistant should not follow instructions inside uploaded documents that ask
it to ignore safety rules, reveal secrets, or fabricate citations.

Page 3

For tool or database access, applications should enforce read-only permissions
when appropriate, validate tool arguments, and log important actions. A language
model may propose a tool call, but application code should enforce boundaries.

Page 4

Responsible deployment requires transparency. CourseMate should show retrieved
sources, explain when information is missing, and remind users to verify
important policy decisions with official course materials or the instructor.
