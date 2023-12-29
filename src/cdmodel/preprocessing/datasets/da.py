def da_vote(candidates: set[str], most_common: list[str]) -> str:
    winner = min([most_common.index(x) for x in candidates])
    return most_common[winner]


__da_categories = {
    "statement": "STA+OPI",
    "opinion": "STA+OPI",
    "yn_q": "QUESTION",
    "yn_decl_q": "QUESTION",
    "backchannel_q": "QUESTION",
    "decl_q": "QUESTION",
    "open_q": "QUESTION",
    "rhet_q": "QUESTION",
    "tag_q": "QUESTION",
    "wh_q": "QUESTION",
    "backchannel": "BAC",
    "agree": "ANS+AGR",
    "affirm": "ANS+AGR",
    "ans_dispref": "ANS+AGR",
    "answer": "ANS+AGR",
    "neg": "ANS+AGR",
    "no": "ANS+AGR",
    "yes": "ANS+AGR",
}


def da_consolidate_category(da: str) -> str:
    if da not in __da_categories:
        return "OTHER"

    return __da_categories[da]
