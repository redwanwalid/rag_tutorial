"""Handle a small set of social messages before document retrieval."""
import re

GREETING = r'(?:hi|hello|hey)(?: there)?|good (?:morning|afternoon|evening)'
WELLBEING = r"how are you(?: doing)?|how's it going|how is it going"


def small_talk_reply(message):
    """Return a canned response only when the entire message is simple small talk.

    All unmatched inputs, including greetings followed by document questions,
    continue through RAG. No model call or similarity threshold decides this route.
    """
    normalized = message.casefold().replace('’', "'")
    normalized = re.sub(r"[^\w\s']", ' ', normalized)
    normalized = ' '.join(normalized.split())
    if re.fullmatch(rf'(?:(?:{GREETING}) )?(?:{WELLBEING})', normalized):
        return "I'm ready to help! What would you like to know about your documents?"
    if re.fullmatch(GREETING, normalized):
        return 'Hi! Ask me a question about your documents, and I’ll look up the relevant passages.'
    if normalized in ('thanks', 'thank you', 'thanks a lot', 'thank you very much'):
        return 'You’re welcome! Let me know if you have another question about your documents.'
    if normalized in ('bye', 'goodbye', 'see you', 'see you later'):
        return 'Goodbye! Come back anytime you have questions about your documents.'
    return None
