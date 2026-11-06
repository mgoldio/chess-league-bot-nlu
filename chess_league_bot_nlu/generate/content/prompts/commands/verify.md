Classify each message sent to a chess league's chat bot. For each, give the single best intent from this list, or "none" if it is not a request the bot can act on (chat, thanks, questions it can't answer, near misses).

Intents:
{intents}

For "{notify}" messages, also say which event they want to hear about: {events}.

Messages:
{messages}

Return only a JSON array with one object per message, in order: {{"intent": "...", "event": one of those events, or null}}
