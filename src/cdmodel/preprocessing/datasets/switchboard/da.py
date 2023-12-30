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


def load_dialogue_acts(
    id: int,
    speaker: str,
    switchboard_dir: str,
) -> list[SwitchboardDialogueAct]:
    """
    Load all Switchboard dialogue acts in a conversation from NXT-Switchboard annotations.

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


def load_terminals(
    id: int,
    speaker: str,
    switchboard_dir: str,
) -> dict[str, SwitchboardTerminal]:
    """
    Load all Switchboard terminals in a conversation from NXT-Switchboard annotations.

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

    terminals: dict[str, SwitchboardTerminal] = {}

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


def pair_terminals_das(
    terminals: dict[str, SwitchboardTerminal], das: list[SwitchboardDialogueAct]
) -> dict[str, tuple[SwitchboardTerminal, SwitchboardDialogueAct]]:
    """
    Pair terminals with their associated dialogue act.

    Parameters
    ----------
    terminals : dict[str, SwitchboardTerminal]
        A dictionary mapping terminal IDs to a SwitchboardTerminal object.
    das : list[SwitchboardDialogueAct]
        A list of SwitchboardDialogueAct objects from a conversation.

    Returns
    -------
    dict[str, tuple[SwitchboardTerminal, SwitchboardDialogueAct]]
        A dictionary mapping terminal ID to a SwitchboardTerminal and its
        associated SwitchboardDialogueAct.

    Raises
    ------
    Exception
        _description_
    """
    paired_terminals_das: dict[
        str, tuple[SwitchboardTerminal, SwitchboardDialogueAct]
    ] = {}

    for da in das:
        for terminal_id in da.terminal_ids:
            # A terminal may not be in the dictionary of terminals if it was omitted for not
            # having a start/end time
            if terminal_id not in terminals:
                continue

            # This shouldn't ever happen. If it does, it means multiple dialogue acts were
            # associated with a turn - investigate if you see this
            if terminal_id in paired_terminals_das:
                raise Exception(
                    f"Terminal {terminal_id} has already been assigned to a dialogue act!"
                )

            # Pair the dialogue act and terminal
            paired_terminals_das[terminal_id] = (terminals[terminal_id], da)

    return paired_terminals_das
