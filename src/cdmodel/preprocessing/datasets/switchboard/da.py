from os import path

from bs4 import BeautifulSoup


def get_dialogue_acts(
    id: int,
    speaker: str,
    base_dir: str,
):
    id_str = f"sw{id}.{speaker}"
    base_dir = path.join(base_dir, "nxt_switchboard_ann", "xml")

    soup = BeautifulSoup(
        open(path.join(base_dir, "dialAct", f"{id_str}.dialAct.xml")), "xml"
    )

    dialogue_acts = []

    for x in soup.find_all("da"):
        dialogue_act = {
            "id": x["nite:id"],
            "nitetype": x["niteType"],
            "swbdtype": x["swbdType"],
            "terminals": [],
        }

        for c in x.children:
            if c.name != "child":
                continue

            dialogue_act["terminals"].append(c["href"])

        dialogue_acts.append(dialogue_act)

    return dialogue_acts


def get_terminals(
    id: int,
    speaker: str,
    base_dir: str,
):
    id_str = f"sw{id}.{speaker}"
    base_dir = path.join(base_dir, "nxt_switchboard_ann", "xml")

    terminals = {}

    soup = BeautifulSoup(
        open(path.join(base_dir, "terminals", f"{id_str}.terminals.xml")), "xml"
    )

    for w in soup.find_all("word"):
        try:
            start = float(w["nite:start"])
            end = float(w["nite:start"])
        except:
            continue

        terminals[f"sw{id}.{speaker}.terminals.xml#id({w['nite:id']})"] = {
            "start": start,
            "end": end,
            "orth": w["orth"],
        }

    return terminals


def expand_terminals_da(dialogue_acts: list[dict], terminals: list[dict]) -> None:
    for d in dialogue_acts:
        new_terminals = []

        for t in d["terminals"]:
            if t not in terminals:
                continue

            t = terminals[t]
            new_terminals.append(t)

            if "swbdtype" in t:
                raise Exception("Already assigned!")

            t["swbdtype"] = d["swbdtype"]
            t["nitetype"] = d["nitetype"]
        d["terminals"] = new_terminals
