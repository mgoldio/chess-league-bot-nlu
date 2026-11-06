from chess_league_bot_nlu import dataset, records


def command(record_id: str, text: str, intent: str = "none") -> records.Command:
    return records.Command.from_dict({"id": record_id, "text": text, "intent": intent})


def test_a_command_text_is_kept_once_and_on_one_side_of_the_split():
    texts = ["FAQ?", "faq", "  Faq!! ", "tell me a joke", "Tell me a joke."]
    train, test = dataset.split([command(f"c{index}", text) for index, text in enumerate(texts)])
    kept = [dataset.plain(each.text) for each in train + test]
    assert sorted(kept) == ["faq", "tell me a joke"]


def test_about_the_test_share_of_texts_are_held_out():
    held_out = sum(dataset.is_test(f"message {index}") for index in range(4000)) / 4000
    assert abs(held_out - dataset.TEST_SHARE) < 0.03


def test_a_text_written_for_two_different_commands_is_left_out():
    train, test = dataset.split([command("c1", "pairings", "pairings_link"), command("c2", "Pairings!", "pairing")])
    assert train + test == []


def test_directories_joined_with_a_plus_read_as_one_dataset(tmp_path):
    for name, text in (("a", "faq"), ("b", "help")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "commands.jsonl").write_text(f'{{"id": "{name}", "text": "{text}", "intent": "{text}"}}\n')
    assert [command.id for command in dataset.commands(f"{tmp_path / 'a'}+{tmp_path / 'b'}")] == ["a", "b"]


def test_a_directory_may_carry_the_share_of_its_records_shown_each_epoch(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "commands.jsonl").write_text('{"id": "a1", "text": "faq", "intent": "faq"}\n')
    data = f"{tmp_path / 'a'}@0.5"
    assert [command.id for command in dataset.commands(data)] == ["a1"]
    assert dataset.shares(data) == {"a1": 0.5}
