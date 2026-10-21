"""Render a selection as the text that goes back into the model's context."""

from __future__ import annotations

from .atoms import Atom
from .select import Selection
from .tokens import estimate

REPLACE_HEADER = (
    "# Carried-over context (context-graph)\n"
    "The conversation was cleared. This is what was kept from it: {kept} of {total} items from the "
    "session graph, chosen by relevance to the latest request and by recency, about {tokens} tokens. "
    "Items under PINNED are the user's constraints and the decisions already made, copied word for "
    "word: follow them. To read the full text of any item, run `{ctx} get <id>` with Bash."
)
ASSIST_HEADER = (
    "# Pinned context (context-graph)\n"
    "Kept from before compaction, in addition to the summary. Items under PINNED are the user's "
    "constraints and the decisions already made, copied word for word: follow them even where the "
    "summary paraphrases or omits them. To read the full text of any item, run `{ctx} get <id>` with Bash."
)
RECENCY_HEADER = (
    "# Carried-over context (last part of the conversation)\n"
    "The conversation was cleared. This is the most recent part of it, about {tokens} tokens."
)


def header_for(mode: str, selector: str, ctx: str, kept: int = 999, total: int = 9999, tokens: int = 99999) -> str:
    if selector == "recency":
        return RECENCY_HEADER.format(tokens=tokens)
    tpl = ASSIST_HEADER if mode == "assist" else REPLACE_HEADER
    return tpl.format(kept=kept, total=total, tokens=tokens, ctx=ctx)


def header_tokens(mode: str, selector: str, ctx: str, cpt: float) -> int:
    # sized with the widest numbers, plus the section headings
    return estimate(header_for(mode, selector, ctx), cpt) + 30


def _line(a: Atom, with_source: bool = False) -> str:
    where = f"turn {a.turn}, {a.source}" if with_source else f"turn {a.turn}"
    return f"- [{a.id}] ({where}) {a.render}"


def render(sel: Selection, atoms: list[Atom], mode: str = "replace", ctx: str = "ctx", cpt: float = 3.5) -> str:
    by_id = {a.id: a for a in atoms}
    kept = [by_id[i] for i in sel.kept if i in by_id]
    kept.sort(key=lambda a: (a.seq, a.id))
    latest = f"u{sel.now_turn}"

    body: list[str] = []
    if sel.selector == "recency":
        body.append("## Conversation tail")
        body += [_line(a) for a in kept]
    else:
        pins = [a for a in kept if a.pinned]
        rest = [a for a in kept if not a.pinned and a.id != latest]
        if pins:
            body.append("## PINNED: constraints and decisions (verbatim)")
            body += [_line(a, with_source=True) for a in pins]
        if rest:
            body.append("## Session record (most relevant items, oldest first)")
            body += [_line(a) for a in rest]
        if latest in by_id and latest in sel.kept:
            body.append("## Latest request")
            body.append(_line(by_id[latest]))

    text_body = "\n".join(body)
    tokens = estimate(text_body, cpt)
    head = header_for(mode, sel.selector, ctx, kept=len(kept), total=sel.n_atoms, tokens=tokens)
    return head + "\n\n" + text_body + "\n"
