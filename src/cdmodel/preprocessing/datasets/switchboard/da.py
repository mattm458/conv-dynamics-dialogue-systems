from os import path
from typing import NamedTuple, Optional

from bs4 import BeautifulSoup


class SwitchboardTerminal(NamedTuple):
    start: float
    end: float
    orth: str


class SwitchboardDialogueAct(NamedTuple):
    id: str
    nitetype: str
    swbdtype: str
    terminal_ids: list[str]


class SwitchboardDialogueActTerminal(NamedTuple):
    terminal: SwitchboardTerminal
    dialogue_acts: list[SwitchboardDialogueAct]


def load_dialogue_acts(
    id: str,
    speaker: str,
    switchboard_dir: str,
) -> list[SwitchboardDialogueAct]:
    """
    Load all Switchboard dialogue acts in a conversation from Switchboard-NXT annotations.

    Parameters
    ----------
    id : int
        The Switchboard conversation ID.
    speaker : str
        The Switchboard speaker, either `A` or `B`.
    switchboard_dir : str
        The Switchboard base directory.

    Returns
    -------
    list[SwitchboardDialogueAct]
        A list of dialogue acts in the conversation.

    Raises
    ------
    FileNotFoundError
        Raised if NXT-Switchboard annotations are not available in the given Switchboard directory.
    """
    id_str = f"sw{id}.{speaker}"
    base_dir = path.join(switchboard_dir, "nxt_switchboard_ann", "xml")

    if not path.exists(base_dir):
        raise FileNotFoundError(
            f"NXT-Switchboard XML annotations not available at {base_dir}!"
        )

    soup = BeautifulSoup(
        open(path.join(base_dir, "dialAct", f"{id_str}.dialAct.xml")), "xml"
    )

    dialogue_acts = []

    for x in soup.find_all("da"):
        terminal_ids: list[str] = []

        for c in x.children:
            if c.name != "child":
                continue

            terminal_ids.append(c["href"])

        dialogue_acts.append(
            SwitchboardDialogueAct(
                id=x["nite:id"],
                nitetype=x["niteType"],
                swbdtype=x["swbdType"],
                terminal_ids=terminal_ids,
            )
        )

    return dialogue_acts


def get_terminals(
    id: int,
    speaker: str,
    switchboard_dir: str,
) -> dict[str, SwitchboardTerminal]:
    """
    Load all Switchboard dialogue acts in a conversation from Switchboard-NXT annotations.

    Parameters
    ----------
    id : int
        The Switchboard conversation ID.
    speaker : str
        The Switchboard speaker, either `A` or `B`.
    switchboard_dir : str
        The Switchboard base directory.

    Returns
    -------
    dict[str, SwitchboardTerminal]
        A dictionary mapping the terminal ID to a SwitchboardTerminal object.

    Raises
    ------
    FileNotFoundError
        Raised if NXT-Switchboard annotations are not available in the given Switchboard directory.
    """
    id_str = f"sw{id}.{speaker}"
    base_dir = path.join(switchboard_dir, "nxt_switchboard_ann", "xml")

    if not path.exists(base_dir):
        raise FileNotFoundError(
            f"NXT-Switchboard XML annotations not available at {base_dir}!"
        )

    terminals = {}

    soup = BeautifulSoup(
        open(path.join(base_dir, "terminals", f"{id_str}.terminals.xml")), "xml"
    )

    for w in soup.find_all("word"):
        # Some NXT-Switchboard terminals do not have valid start or end times.
        # This usually happens for short, unvoiced, or noise-related terminals.
        # If that is the case for this terminal, then omit it from output.
        try:
            start = float(w["nite:start"])
            end = float(w["nite:start"])
        except:
            continue

        terminal_id = f"sw{id}.{speaker}.terminals.xml#id({w['nite:id']})"
        terminals[terminal_id] = SwitchboardTerminal(
            start=start, end=end, orth=w["orth"]
        )

    return terminals


def expand_terminals_da(
    das: list[SwitchboardDialogueAct], terminals: dict[str, SwitchboardTerminal]
) -> dict[str, SwitchboardDialogueActTerminal]:
    expanded_terminals: dict[str, SwitchboardDialogueActTerminal] = dict(
        [
            (
                terminal_id,
                SwitchboardDialogueActTerminal(terminal=terminal, dialogue_acts=[]),
            )
            for (terminal_id, terminal) in terminals.items()
        ]
    )

    for da in das:
        for terminal_id in da.terminal_ids:
            if terminal_id not in expanded_terminals:
                continue

            expanded_terminals[terminal_id].dialogue_acts.append(da)

    return expanded_terminals
