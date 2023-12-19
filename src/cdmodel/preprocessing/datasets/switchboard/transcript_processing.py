import re
from pandas import DataFrame

## PROCESSING CODE FROM https://github.com/emeinhardt/switchboard-lm/
interrupted_word_pattern = ".*-$"
resumed_word_pattern = "^-.*"


def isInterrupted(wordform):
    return re.match(interrupted_word_pattern, wordform) is not None


def isResumed(wordform):
    return re.match(resumed_word_pattern, wordform) is not None


def isBroken(wordform):
    return isInterrupted(wordform) or isResumed(wordform)


def hasBrokenWords(speech):
    speech_word_seq = speech.split(" ")
    broken_words = list(filter(isBroken, speech_word_seq))
    return len(broken_words) > 0


def remove_broken_words(speech, insertUnk=False):
    if insertUnk:
        replacement = unk
    else:
        replacement = ""

    speech_word_seq = speech.split(" ")
    speech_word_seq_fixed = " ".join(replace(speech_word_seq, isBroken, (replacement,)))
    speech_fixed = " ".join([w for w in speech_word_seq_fixed.split(" ") if len(w) > 0])
    return speech_fixed


def hasCurlyBraces(wordform):
    return "{" in wordform or "}" in wordform


def isCurlyBraced(wordform):
    if len(wordform) == 0:
        return False
    return wordform[0] == "{" and wordform[-1] == "}"


def removeCurlyBraces(wordform):
    if not isCurlyBraced(wordform):
        return wordform
    return wordform[1:-1]


def remove_curly_braces(speech):
    speech_word_seq = speech.split(" ")
    speech_word_seq_fixed = " ".join(list(map(removeCurlyBraces, speech_word_seq)))
    speech_fixed = " ".join([w for w in speech_word_seq_fixed.split(" ") if len(w) > 0])
    return speech_fixed


def hasUnderscore(wordform):
    return "_" in wordform


def fixUnderscore(wordform):
    if not hasUnderscore(wordform):
        return wordform
    fixed = wordform.replace("_1", "")
    return fixed


def fix_underscores(speech):
    speech_word_seq = speech.split(" ")
    speech_word_seq_filtered = [
        w for w in speech_word_seq if w != "<b_aside>" and w != "<e_aside>"
    ]
    speech_word_seq_fixed = " ".join(list(map(fixUnderscore, speech_word_seq_filtered)))
    speech_fixed = " ".join([w for w in speech_word_seq_fixed.split(" ") if len(w) > 0])
    return speech_fixed


def lowercase(speech):
    return speech.lower()


def removeNonSpeech(word):
    if len(word) > 0 and word[0] != "[" and word[-1] != "]":
        return word
    else:
        return None


def process_word(word):
    word = lowercase(word)
    word = fixUnderscore(word)
    word = remove_curly_braces(word)

    if isBroken(word):
        return None

    word = removeNonSpeech(word)
    return word
