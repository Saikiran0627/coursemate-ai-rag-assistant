# Module 3: Agents and Tool Use

An AI agent is a system that uses a language model to decide what action to take
toward a goal. A tool-using agent can call external capabilities such as a
calculator, search API, file reader, database, or Python function. The agent
observes the tool result and uses it to decide the next step or produce the
final answer.

Tool use is helpful when a task requires information or computation that the
model should not guess. For example, arithmetic should be delegated to a
calculator, current facts should be delegated to search, and private structured
records should be delegated to a database with safe access controls.

Agent systems should log tool calls, validate tool arguments, and restrict
dangerous actions. A language model can propose an action, but the application
code should enforce permissions, rate limits, and read-only access when
appropriate.

Evaluation of agents should include tasks where no tool is needed, tasks where
one tool is clearly needed, and tasks where the agent must choose between
multiple possible tools.
