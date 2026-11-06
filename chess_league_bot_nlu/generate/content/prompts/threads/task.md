League: {league}, round {round}. The pairing was posted {published} UTC; the game must be played before {deadline} UTC.

Plan, one line per message:
{plan}

Write the {count} messages. Rules:
- Where the plan shows a placeholder like {{T1}}, write the placeholder itself, exactly once, where that time goes in the sentence. Never write the time out yourself; it is filled in afterwards. The same goes for {{LIST}} and {{OPPONENT}} where the plan asks for them.
- This holds for agreeing too: write the placeholder the plan gives for that message ("ok {{T5}} works for me"), never the day or time itself. Use only the placeholders given for that message; where a message has none, write no placeholder at all ("ok works for me").
- Apart from placeholders, no message may contain a day or clock time: no weekday names, dates, "tonight" or "tomorrow". Parts of the day ("morning", "late evening"), UTC offsets like +11, and durations like "5 mins" are fine.
- To refer to times from earlier messages that aren't among this message's placeholders, say "those times", "that one" or "it".
- Keep each message within its word limit (placeholders don't count). Messages differ in length: some are a word or two, some run over several lines.
- A message may contain line breaks (\n), and a blank line (\n\n) between paragraphs where the plan says so.
- Avoid assistant-sounding phrasing: no "I would be delighted", "perfectly", "at your earliest convenience", "please let me know", no sign-offs, no greeting in every message, no restating what the other person said.
- The players call each other "you", or nothing. Never invent names, and never @-mention anyone except by writing {{OPPONENT}} where the plan asks.
- English only. No narration or stage directions.

Return only a JSON array of {count} strings, one per message, in order.
