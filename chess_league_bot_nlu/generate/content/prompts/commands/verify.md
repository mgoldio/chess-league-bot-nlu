A message was sent to a chess league's chat bot. Say what it asks the bot for: the single best intent from this list, or "none" if it asks for nothing the bot does (chat, thanks, opinions, things the bot can't do).

People write to the bot the way they type in chat: very short, sloppy, misspelled, with no verb at all. A few words that name what an intent is about are a request for it ("count me in rd 3" says they can play, "skip round 2" says they can't, "pairings" asks for the pairings page, "stop 2" removes subscription 2). Judge what the person most likely wants, not whether they asked politely or completely. Choose "none" only when the message asks the bot for nothing on the list.

Intents, each with how messages for it often look:
{intents}

For "{notify}" messages, also say which event they want to hear about: {events}, or null if they don't say.

Under "also", list any other intent (or "none") a reasonable person could take the message to mean. Leave it empty when the message clearly means one thing.

{arguments}Message: {message}

Return only JSON: {{"intent": "...", "event": one of those events or null, "also": []{argument_reply}}}
