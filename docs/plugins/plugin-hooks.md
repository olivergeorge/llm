(plugin-hooks)=
# Plugin hooks

Plugins use **plugin hooks** to customize LLM's behavior. These hooks are powered by the [Pluggy plugin system](https://pluggy.readthedocs.io/).

Each plugin can implement one or more hooks using the @hookimpl decorator against one of the hook function names described on this page.

LLM imitates the Datasette plugin system. The [Datasette plugin documentation](https://docs.datasette.io/en/stable/writing_plugins.html) describes how plugins work.

(plugin-hooks-register-commands)=
## register_commands(cli)

This hook adds new commands to the `llm` CLI tool - for example `llm extra-command`.

This example plugin adds a new `hello-world` command that prints "Hello world!":

```python
from llm import hookimpl
import click

@hookimpl
def register_commands(cli):
    @cli.command(name="hello-world")
    def hello_world():
        "Print hello world"
        click.echo("Hello world!")
```
This new command will be added to `llm --help` and can be run using `llm hello-world`.

(plugin-hooks-register-models)=
## register_models(register, model_aliases)

This hook can be used to register one or more additional models.

```python
import llm

@llm.hookimpl
def register_models(register):
    register(HelloWorld())

class HelloWorld(llm.Model):
    model_id = "helloworld"

    def execute(self, prompt, stream, response):
        return ["hello world"]
```
If your model includes an async version, you can register that too:

```python
class AsyncHelloWorld(llm.AsyncModel):
    model_id = "helloworld"

    async def execute(self, prompt, stream, response):
        return ["hello world"]

@llm.hookimpl
def register_models(register):
    register(HelloWorld(), AsyncHelloWorld(), aliases=("hw",))
```
This demonstrates how to register a model with both sync and async versions, and how to specify an alias for that model.

The `model_aliases` parameter is a list of {class}`~llm.ModelWithAliases` objects representing all models registered so far by other plugins. Plugins that use `@llm.hookimpl(trylast=True)` can use this to inspect or modify models registered by other plugins. Both parameters are optional - plugins can accept just `register`, just `model_aliases`, or both.

The {ref}`model plugin tutorial <tutorial-model-plugin>` describes how to use this hook in detail. Asynchronous models {ref}`are described here <advanced-model-plugins-async>`.

```{eval-rst}
.. autoclass:: llm.ModelWithAliases
   :exclude-members: matches
```

(plugin-hooks-register-embedding-models)=
## register_embedding_models(register)

This hook can be used to register one or more additional embedding models, as described in {ref}`embeddings-writing-plugins`.

```python
import llm

@llm.hookimpl
def register_embedding_models(register):
    register(HelloWorld())

class HelloWorld(llm.EmbeddingModel):
    model_id = "helloworld"

    def embed_batch(self, items):
        return [[1, 2, 3], [4, 5, 6]]
```

(plugin-hooks-register-tools)=
## register_tools(register)

This hook can register one or more tool functions for use with LLM. See {ref}`the tools documentation <tools>` for more details.

This example registers two tools: `upper` and `count_character_in_word`.

```python
import llm

def upper(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

def count_char(text: str, character: str) -> int:
    """Count the number of occurrences of a character in a word."""
    return text.count(character)

@llm.hookimpl
def register_tools(register):
    register(upper)
    # Here the name= argument is used to specify a different name for the tool:
    register(count_char, name="count_character_in_word")
```

Tools can also be implemented as classes, as described in {ref}`Toolbox classes <python-api-toolbox>` in the Python API documentation.

You can register classes like the `Memory` example {ref}`from here <python-api-toolbox>` by passing the class (_not_ an instance of the class) to `register()`:

```python
import llm

class Memory(llm.Toolbox):
    # Copy implementation from the Python API documentation

@llm.hookimpl
def register_tools(register):
    register(Memory)
```
Once installed, this tool can be used like so:

```bash
llm chat -T Memory
```
If a tool name starts with a capital letter it is assumed to be a toolbox class, not a regular tool function.

Here's an example session with the Memory tool:
```
Chatting with gpt-4.1-mini
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
Type '!edit' to open your default editor and modify the prompt
Type '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments
> Remember my name is Henry

Tool call: Memory_set({'key': 'user_name', 'value': 'Henry'})
  null

Got it, Henry! I'll remember your name. How can I assist you today?
> what keys are there?

Tool call: Memory_keys({})
  [
    "user_name"
  ]

Currently, there is one key stored: "user_name". Would you like to add or retrieve any information?
> read it

Tool call: Memory_get({'key': 'user_name'})
  Henry

The value stored under the key "user_name" is Henry. Is there anything else you'd like to do?
> add Barrett to it

Tool call: Memory_append({'key': 'user_name', 'value': 'Barrett'})
  null

I have added "Barrett" to the key "user_name". If you want, I can now show you the updated value.
> show value

Tool call: Memory_get({'key': 'user_name'})
  Henry
  Barrett

The value stored under the key "user_name" is now:
Henry
Barrett

Is there anything else you would like to do?
```

(plugin-hooks-register-template-loaders)=
## register_template_loaders(register)

Plugins can register new {ref}`template loaders <prompt-templates-loaders>` using the `register_template_loaders` hook.

Template loaders work with the `llm -t prefix:name` syntax. The prefix specifies the loader, then the registered loader function is called with the name as an argument. The loader function should return an `llm.Template()` object.

This example plugin registers `my-prefix` as a new template loader. Once installed it can be used like this:

```bash
llm -t my-prefix:my-template
```
Here's the Python code:

```python
import llm

@llm.hookimpl
def register_template_loaders(register):
    register("my-prefix", my_template_loader)

def my_template_loader(template_path: str) -> llm.Template:
    """
    Documentation for the template loader goes here. It will be displayed
    when users run the 'llm templates loaders' command.
    """
    try:
        # Your logic to fetch the template content
        # This is just an example:
        prompt = "This is a sample prompt for {}".format(template_path)
        system = "You are an assistant specialized in {}".format(template_path)

        # Return a Template object with the required fields
        return llm.Template(
            name=template_path,
            prompt=prompt,
            system=system,
        )
    except Exception as e:
        # Raise a ValueError with a clear message if the template cannot be found
        raise ValueError(f"Template '{template_path}' could not be loaded: {str(e)}")
```
The `llm.Template` class has the following constructor:

```{eval-rst}
.. autoclass:: llm.Template
```

The loader function should raise a `ValueError` if the template cannot be found or loaded correctly, providing a clear error message.

Note that `functions:` provided by templates using this plugin hook will not be made available, to avoid the risk of plugin hooks that load templates from remote sources introducing arbitrary code execution vulnerabilities.

(plugin-hooks-register-fragment-loaders)=
## register_fragment_loaders(register)

Plugins can register new fragment loaders using the `register_template_loaders` hook. These can then be used with the `llm -f prefix:argument` syntax.

Fragment loader plugins differ from template loader plugins in that you can stack more than one fragment loader call together in the same prompt.

A fragment loader can return one or more string fragments or attachments, or a mixture of the two. The fragments will be concatenated together into the prompt string, while any attachments will be added to the list of attachments to be sent to the model.

The `prefix` specifies the loader. The `argument` will be passed to that registered callback..

The callback works in a very similar way to template loaders, but returns either a single `llm.Fragment`, a list of `llm.Fragment` objects, a single `llm.Attachment`, or a list that can mix `llm.Attachment` and `llm.Fragment` objects.

The `llm.Fragment` constructor takes a required string argument (the content of the fragment) and an optional second `source` argument, which is a string that may be displayed as debug information. For files this is a path and for URLs it is a URL. Your plugin can use anything you like for the `source` value.

See {ref}`the Python API documentation for attachments <python-api-attachments>` for details of the `llm.Attachment` class.

Here is some example code:

```python
import llm

@llm.hookimpl
def register_fragment_loaders(register):
    register("my-fragments", my_fragment_loader)


def my_fragment_loader(argument: str) -> llm.Fragment:
    """
    Documentation for the fragment loader goes here. It will be displayed
    when users run the 'llm fragments loaders' command.
    """
    try:
        fragment = "Fragment content for {}".format(argument)
        source = "my-fragments:{}".format(argument)
        return llm.Fragment(fragment, source)
    except Exception as ex:
        # Raise a ValueError with a clear message if the fragment cannot be loaded
        raise ValueError(
            f"Fragment 'my-fragments:{argument}' could not be loaded: {str(ex)}"
        )

# Or for the case where you want to return multiple fragments and attachments:
def my_fragment_loader(argument: str) -> list[llm.Fragment]:
    "Docs go here."
    return [
        llm.Fragment("Fragment 1 content", "my-fragments:{argument}"),
        llm.Fragment("Fragment 2 content", "my-fragments:{argument}"),
        llm.Attachment(path="/path/to/image.png"),
    ]
```
A plugin like this one can be called like so:
```bash
llm -f my-fragments:argument
```
If multiple fragments are returned they will be used as if the user passed multiple `-f X` arguments to the command.

Multiple fragments are particularly useful for things like plugins that return every file in a directory. If these were concatenated together by the plugin, a change to a single file would invalidate the de-duplicatino cache for that whole fragment. Giving each file its own fragment means we can avoid storing multiple copies of that full collection if only a single file has changed.

(plugin-hooks-register-replay-stores)=
## register_replay_stores(register)

This hook registers one or more **replay stores** — objects that can serve a
previously-recorded model response instead of running a live API call. It is
the integration point used by plugins such as `llm-replay` to offer opt-in
record-and-replay for LLM responses in the VCR.py tradition.

Enablement is a plugin concern. `llm` core calls every registered store's
`lookup` once per response iteration; the store decides whether to engage
based on its own state — a CLI flag the plugin registered (e.g. `--replay`),
an environment variable, a module-level toggle set by test fixtures, and so
on. A store that is "off" simply returns `None` from `lookup`, and the live
execute path runs unchanged. `llm-replay` is the canonical example: its
CLI hookimpl flips a module-global when `--replay` is passed, and its
`lookup()` consults that flag before computing a key.

A replay store is a duck-typed object with a small protocol. The synchronous
shape is:

```python
class ReplayStore:
    def lookup(self, response):
        """Return a ReplayedResponse (hit) or None (miss)."""

    def store(self, response):
        """Optionally called by the plugin after a miss to record the live response."""
```

For async responses, the store may also expose `alookup(response)` (awaited by
`AsyncResponse`). If `alookup` is not defined the sync `lookup` is used. A
`ReplayedResponse` is any object with these attributes:

- `chunks: list[str]` — the recorded text chunks to yield
- `response_json: Optional[dict]` — the raw JSON body, if any
- `source_id: Optional[str]` — an identifier the plugin can use to point back
  at the source `responses` row for audit
- `tool_calls: Optional[list[llm.ToolCall]]` — tool calls the model emitted on
  the recorded run. Populated onto `response._tool_calls` so the chain can
  continue without re-querying the model.
- `tool_results: Optional[list[llm.ToolResult]]` — recorded outputs of the
  tools that fired during the initial recording. When provided,
  `execute_tool_calls()` returns this list verbatim instead of invoking live
  tool implementations, so destructive tools only fire once per recording.
- `resolved_model: Optional[str]` — the provider-resolved model id from the
  recorded run. Written onto `response.resolved_model` so replays don't
  degrade the audit trail.

The `response` argument to `lookup` is the active `llm.Response` (or
`llm.AsyncResponse`); the plugin reads `response.prompt`, `response.model`,
and `response.conversation` to compute its own replay key. Any additional
plugin-specific inputs (e.g. whether URL attachments should be fetched and
content-hashed) live in plugin state, not on the response object. This keeps
the request-representation dataclass private to the plugin, so changes to how
a plugin canonicalizes a request do not force a change to `llm` core.

When multiple stores are registered, they are consulted in dispatch order and
the first non-`None` `lookup` result wins. On a hit, `_BaseResponse` populates
the chunks, marks the response done, records `response.replayed = True` and
`response.replay_source_id`, and skips `model.execute(...)` entirely.

Here is a minimal stub plugin useful for tests:

```python
import llm

class StubStore:
    def __init__(self, replayed=None):
        self.replayed = replayed

    def lookup(self, response):
        return self.replayed

class ReplayedResponse:
    def __init__(self, chunks, response_json=None, source_id=None):
        self.chunks = chunks
        self.response_json = response_json
        self.source_id = source_id

@llm.hookimpl
def register_replay_stores(register):
    register(StubStore(replayed=ReplayedResponse(["hello from cache"])))
```

With this plugin installed, `model.prompt("anything").text()` returns
`"hello from cache"` without touching the upstream API. A production plugin
would gate that return value on a flag it owns so the store only engages
when the user has opted in.

The hookspec is **provisional** for at least one release cycle. Third-party
stores can experiment against it, but the signature may change while we live
with the contract.

(plugin-hooks-register-prompt-gates)=
## register_prompt_gates(register)

This hook registers one or more **prompt gates** — objects that can veto a
prompt before `model.execute` is called. It is the integration point used by
plugins such as `llm-confirm-tokens` to interpose a "you are about to send N
tokens, proceed?" confirmation on top of the normal prompt flow without
wrapping the `llm` CLI itself.

Enablement is the gate's concern. Core calls every registered gate's
`check` once, in pluggy dispatch order, immediately before the first chunk
is requested from the model. A gate that is "off" (e.g. stdin isn't a TTY,
the token count is under a threshold, the user passed `--yes`) simply
returns `None` and the live execute path runs unchanged.

A gate is a duck-typed object with a small protocol:

```python
class PromptGate:
    def check(self, prompt, model, conversation=None):
        """Return None to allow the prompt, or raise llm.CancelPrompt to abort."""
```

For async responses, a gate may also expose `acheck(prompt, model,
conversation=None)` (awaited by `AsyncResponse`). If `acheck` is not defined
the sync `check` is used.

Arguments:

- `prompt` is the fully-resolved `llm.Prompt` that is about to be sent —
  fragments, attachments, system prompt, tools and options are all already
  populated, so a gate that wants to count tokens or audit the final
  message list can read them directly.
- `model` is the `llm.Model` or `llm.AsyncModel` that will execute the
  prompt.
- `conversation` is the `llm.Conversation` the prompt belongs to, or `None`
  for one-shot prompts. When `llm -c`/`--cid` continues a prior session,
  `conversation.responses` holds the earlier turns that the model will
  re-send alongside the new prompt — gates that care about what the
  provider actually bills (token counters, size caps) should walk it.

Core calls `check` with `conversation` as a keyword argument. Gates that
omit it from their signature are invoked in the legacy `(prompt, model)`
shape instead, so pre-existing gates keep working — they just won't see
history.

To cancel the prompt, raise `llm.CancelPrompt("reason")`. The exception
propagates to the caller; no chunks are yielded, the conversation is not
updated, and `model.execute` is not invoked. Because cancellation is a
raised exception rather than a return value, the first raising gate
short-circuits any subsequent gates.

Gates do **not** fire when a replay store has already satisfied the response
locally — the replay short-circuit runs first, so a gate sees the prompt
only when a real upstream API call is about to happen.

Here is a minimal stub plugin useful for tests:

```python
import llm

class BlockPrompts:
    def check(self, prompt, model):
        raise llm.CancelPrompt("prompts are disabled in this environment")

@llm.hookimpl
def register_prompt_gates(register):
    register(BlockPrompts())
```

With this plugin installed, `model.prompt("anything").text()` raises
`llm.CancelPrompt` before touching the upstream API. A production plugin
would gate the `raise` on its own state — a threshold, a CLI flag, a TTY
check — so the gate only engages when the user has opted in.

The hookspec is **provisional** for at least one release cycle. Third-party
gates can experiment against it, but the signature may change while we
live with the contract.

(plugin-hooks-after-log-to-db)=
## after_log_to_db(response, db)

This hook fires after `Response.log_to_db` has persisted a response to the
logs database. Plugins use it to record auxiliary metadata keyed on the
now-logged response — for example indexes, audit trails, or cross-response
hashes.

Gating on `log_to_db` itself means plugins inherit `llm`'s existing log
policy (`logs_on()`, `--log` / `--no-log`) for free: the hook fires if and
only if `llm` decided to log the response.

The `response` argument is the `Response` (or `AsyncResponse`) that was just
persisted, and `db` is the `sqlite_utils.Database` it was written to. The
row is already present at hook time, so plugins can read it back.

Example:

```python
import llm
import sqlite_utils

@llm.hookimpl
def after_log_to_db(response: llm.Response, db: sqlite_utils.Database):
    db["audit"].insert({
        "response_id": response.id,
        "model": response.model.model_id,
    })
```
