from song_recommender.features.lyrics import safe_lyrics


def test_safe_lyrics_none():
    assert safe_lyrics(None) == ""

