> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Field Manual Reference Pages

These pages are lookup aids for use while reading source, designing a scenario, or reviewing a result. They compress the longer chapters; when a summary and a chapter appear to disagree, follow the chapter's cited source and the authority order in the [manual index](../README.md).

| Reference | Use it when |
|---|---|
| [Example tree and asset map](example_tree_and_asset_map.md) | You need to understand the major example architectures, `Support/`, `dataForExamples/`, or how a static/binary asset enters a simulation. |
| [Example capability index](example_capability_index.md) | You have an engineering capability in mind and need the smallest useful local examples, important modules, and main caveats. |
| [Module and message glossary](module_and_message_glossary.md) | You encounter a Basilisk class, scheduler concept, message role, frame symbol, or repository architecture family. |
| [Frame, unit, initialization, and validation checklists](frame_unit_and_initialization_checklists.md) | You are designing, debugging, or reviewing a simulation and want a pre-run or pre-plot challenge list. |

These are not API inventories. Exact payload fields, required inputs, defaults, and version behavior remain the responsibility of the installed Basilisk source and version-matched official documentation.

When vetting a reference entry, record:

```text
Entry or table row:
Source file/API checked:
Basilisk version:
Observed correction:
Engineering consequence:
Related chapter(s) to update:
Reviewer and date:
```

Update both the reference entry and its longer chapter when a correction changes meaning. This prevents a quick lookup page from drifting away from the evidence behind it.
