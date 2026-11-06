"""A causal language model's trunk with small fixed heads, in place of its word head.

A thread is read left to right, so a message's answers depend only on the
messages up to it. Times are tagged on their tokens, each tag looking a few
tokens ahead within its message. At a message's last token, one pointer head
answers for every time written so far what this message does with it: offers
it, restates it while accepting, accepts it, takes it back, mentions it as
something other than an offer, or nothing. Which player could have done what
is decided in code (`outcome`). Commands share the trunk: the
command and its event are read at the last token, and arguments are tagged.
"""
