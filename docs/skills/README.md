# Editable project skills

These are the maintained, editable copies of the Voyager architecture and target
adoption skills. They contain target-agnostic instructions and relative links to
this checkout's shared docs. The default target is only an optional example.
Start model work at [Create and register the model](../compilation/performance-models.md#create-and-register-the-model).

## Install in an isolated checkout

Keep this directory and the shared docs with the isolated compiler tree. From
that checkout's root, using an existing Python 3.9+ environment:

```bash
python docs/skills/install.py --dry-run
python docs/skills/install.py
```

To select a different destination checkout, add `--repo PATH`. Its shared docs
must be present and match the packaged contracts. The installer checks relative
link destinations before copying both skills into:

```text
.agents/skills/
  voyager-compiler-architecture/
    SKILL.md
    agents/openai.yaml
  voyager-target-adoption/
    SKILL.md
    agents/openai.yaml
    references/model-validation.md
```

Use an agent session configured to discover these project skills, or explicitly
point it to the installed `SKILL.md`. Copying files does not reload an existing
session or isolate it from globally installed skills. No network, new environment,
global skill modification or compiler change is required by this installer.

## Edit and refresh

Edit [architecture](voyager-compiler-architecture/SKILL.md) or
[target adoption](voyager-target-adoption/SKILL.md) here. Installed skills are
independent copies, so experiment-specific edits there do not change this source.
The installer refuses existing destinations by default. To refresh from this
source, use `--replace`; it replaces the two installed skill directories,
including local edits. Bring wanted edits back to this source before refreshing.

Links such as `../../../docs/roles.md` work from both `docs/skills/<name>/` and
`.agents/skills/<name>/`, because both are three levels below the checkout root.
Internal reference links remain relative to their skill. Do not copy these
packages into an arbitrary global directory and expect checkout links to resolve.

## Experiment boundary

Choose the allowed compiler/docs/specification inputs before starting the agent.
Install these packages in that tree; do not let the agent use global skill copies
that route to donor checkouts. Historical reports, previous target implementations,
profiles and repository history can reveal the answer. Package/access isolation
is a separate evaluator responsibility; this installer only copies the skills.
The [adoption route](../adoption.md) explains implementation work, not how to
remove all target dependencies from the compiler.
