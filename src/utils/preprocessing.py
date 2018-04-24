import json
import re
import sys

import nltk

import app_data as app_data
from src.utils.language_detect import LanguageDetector


def preprocess_lyrics():

    metal_subgenres_lyrics_map = _preprocess_metal_lyrics()

    _save_preprocessed_lyrics(metal_subgenres_lyrics_map, metal_only=True)

    other_genres_lyrics_map = _preprocess_other_lyrics()

    # Join all metal lyrics in one array, add it to other genres map and
    # save it in one preprocessed file
    metal_lyrics = []
    for genre, lyrics in metal_subgenres_lyrics_map.items():
        metal_lyrics = metal_lyrics + lyrics

    genre_lyrics_map = other_genres_lyrics_map
    genre_lyrics_map["metal"] = metal_lyrics

    # All genres finished, save genre map of preprocessed lyrics
    _save_preprocessed_lyrics(genre_lyrics_map, metal_only=False)


def _preprocess_metal_lyrics():

    # Get available metal subgenres
    lyrics_genres = app_data.LYRICS_GENRES_METAL

    # If app is in debug mode, use test files
    if app_data.DEBUG_MODE:
        lyrics_genres_file_names = [genre + "_test.txt" for genre in lyrics_genres]
    else:
        lyrics_genres_file_names = [genre + ".txt" for genre in lyrics_genres]

    genre_lyrics_map = {}

    for index, genre in enumerate(lyrics_genres):

        file_name = lyrics_genres_file_names[index]
        file_path = _get_raw_lyrics_file_path(file_name, genre_type="metal")

        processed_lyrics = []

        skipped_words_map = {}
        skipped_artists_count = 0
        invalid_lyrics_count = 0

        # Open the file containing the current genre's lyrics
        with open(file_path, "rt", encoding="utf-8") as file:

            # Get content from file
            content = json.loads(file.read())

            print("Got JSON from file " + file_name)

            # Check if file content is a JSON list
            json_is_list = type(content) is list

            if json_is_list:

                print("[✔] \t JSON is in list format")

                # Go through list of artist JSONs and process them
                for artist_info in content:

                    artist_genre = artist_info["artistGenre"]
                    album_lyrics_str = artist_info["albumLyrics"]

                    allowed_check_result = _is_subgenre_allowed(genre, artist_genre)
                    allowed = allowed_check_result[0]

                    # Skip this artist if his genre is not allowed
                    if not allowed:
                        genre_word = allowed_check_result[1]
                        skipped_words_map[genre_word] = skipped_words_map.get(genre_word, 0) + 1
                        skipped_artists_count += 1
                        continue

                    # Split album lyrics by separator to get array of album lyrics pages
                    album_lyrics_arr = album_lyrics_str.split(app_data.LYRICS_SEPARATOR)

                    # Process all album lyrics pages
                    for album_lyrics in album_lyrics_arr:

                        # If this album page contains the "thanks to..." part at the end, cut it out
                        if 'class="thanks"' in album_lyrics:
                            thanks_div_start = album_lyrics.find('<div class="thanks">')
                            album_lyrics = album_lyrics[:thanks_div_start]

                        # Every song's lyrics start with <h3> - get song lyrics array from album lyrics page
                        song_lyrics = album_lyrics.split("<h3>")
                        for lyrics in song_lyrics:

                            # Strings and regex for filtering lyrics
                            unwanted_strings = app_data.PREPROCESS_LYRICS_UNWANTED_STRINGS
                            unwanted_regex = app_data.PREPROCESS_LYRICS_UNWANTED_REGEX

                            lyrics = lyrics.lower()

                            # Remove unwanted strings
                            for string in unwanted_strings:
                                lyrics = lyrics.replace(string, "")

                            # Remove unwanted regex
                            for regex in unwanted_regex:
                                lyrics = re.sub(regex, "", lyrics, flags=re.IGNORECASE)

                            # Remove everything except alphanumeric strings, newlines and apostrophes
                            lyrics = re.sub("[^A-Za-z0-9\\\\\s\']+", '', lyrics)

                            # Skip lyrics if they aren't valid
                            lyrics_valid = _are_lyrics_valid(lyrics)
                            if not lyrics_valid:
                                invalid_lyrics_count += 1
                                continue

                            processed_lyrics.append(lyrics)

            else:
                print("[x] \t Error: Lyrics JSON for genre " + genre + " is not in list format")

        print("Skipped artists count for genre " + genre + ": " + str(skipped_artists_count))
        print("Skipped artists per invalid word for genre " + genre + ": " + str(skipped_words_map))
        print("Skipped invalid lyrics count: " + str(invalid_lyrics_count))

        # Save processed lyrics for current genre
        genre_lyrics_map[genre] = processed_lyrics

    return genre_lyrics_map


def _preprocess_other_lyrics():

    # Get available metal subgenres
    lyrics_genres = app_data.LYRICS_GENRES_OTHER

    # If app is in debug mode, use test files
    if app_data.DEBUG_MODE:
        lyrics_genres_file_names = [genre + "_test.txt" for genre in lyrics_genres]
    else:
        lyrics_genres_file_names = [genre + ".txt" for genre in lyrics_genres]

    genre_lyrics_map = {}

    for index, genre in enumerate(lyrics_genres):

        file_name = lyrics_genres_file_names[index]
        file_path = _get_raw_lyrics_file_path(file_name, genre_type="other")

        processed_lyrics = []

        # Open the file containing the current genre's lyrics
        with open(file_path, "rt", encoding="utf-8") as file:

            # Get content from file
            content = json.loads(file.read())

            print("Got JSON from file " + file_name)

            # Check if file content is a JSON list
            json_is_list = type(content) is list

            if json_is_list:

                print("[✔] \t JSON is in list format")

                # Go through list of artist JSONs and process them
                for lyrics in content:

                    # Strings and regex for filtering lyrics
                    unwanted_strings = app_data.PREPROCESS_LYRICS_UNWANTED_STRINGS
                    unwanted_regex = app_data.PREPROCESS_LYRICS_UNWANTED_REGEX

                    lyrics = lyrics.lower()

                    # Remove unwanted strings
                    for string in unwanted_strings:
                        lyrics = lyrics.replace(string, "")

                    # Remove unwanted regex
                    for regex in unwanted_regex:
                        lyrics = re.sub(regex, "", lyrics, flags=re.IGNORECASE)

                    # Remove everything except alphanumeric strings, newlines and apostrophes
                    lyrics = re.sub("[^A-Za-z0-9\\\\\s\']+", '', lyrics)

                    # Skip lyrics if they aren't valid
                    lyrics_valid = _are_lyrics_valid(lyrics)
                    if not lyrics_valid:
                        continue

                    processed_lyrics.append(lyrics)

            else:
                print("[x] \t Error: Lyrics JSON for genre " + genre + " is not in list format")

        # Save processed lyrics for current genre
        genre_lyrics_map[genre] = processed_lyrics

    return genre_lyrics_map


def _remove_newlines(string):
    result = string.replace("\\n", " ")
    return result


def _get_raw_lyrics_file_path(file_name, genre_type="metal"):

    if genre_type is "metal":
        raw_lyrics_folder_path = app_data.RAW_LYRICS_SUBFOLDER_PATH_METAL
    else:
        raw_lyrics_folder_path = app_data.RAW_LYRICS_SUBFOLDER_PATH_OTHER

    return raw_lyrics_folder_path + "/" + file_name


def _save_preprocessed_lyrics(genre_lyrics_map, metal_only=True):

    if app_data.DEBUG_MODE:
        output_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH_METAL_TEST if metal_only else app_data.PREPROCESSED_LYRICS_FILE_PATH_ALL_TEST
    else:
        output_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH_METAL if metal_only else app_data.PREPROCESSED_LYRICS_FILE_PATH_ALL

    with open(output_file_path, 'w') as output_file:
        json.dump(genre_lyrics_map, output_file)


def _are_lyrics_valid(lyrics):

    min_length = app_data.MIN_LYRICS_LENGTH

    lyrics_are_english = LanguageDetector.is_english(lyrics)
    lyrics_length_ok   = len(lyrics) >= min_length

    return lyrics_are_english and lyrics_length_ok


def get_lyrics_dict(lyrics):

    # Strings and regex for filtering lyrics
    unwanted_strings = app_data.PREPROCESS_LYRICS_UNWANTED_STRINGS
    unwanted_regex = app_data.PREPROCESS_LYRICS_UNWANTED_REGEX

    lyrics = lyrics.lower()

    verse_break = app_data.VERSE_BREAK
    stanza_break = app_data.STANZA_BREAK

    # Count verses and stanzas in song lyrics (add one because breaks are only between verses/stanzas)
    verse_count = lyrics.count(verse_break) + 1
    stanza_count = lyrics.count(stanza_break) + 1

    # Extract average verse length feature
    verses = lyrics.split(verse_break)
    number_of_verses = len(verses)

    total_verses_length = 0
    total_words_count   = 0
    for verse in verses:
        verse_length = len(verse)
        total_verses_length += verse_length

        verse_words = verse.split(" ")
        verse_words_count = len(verse_words)
        total_words_count += verse_words_count

    average_verse_length = total_verses_length / number_of_verses
    average_verse_word_count = total_words_count / number_of_verses

    # Remove unwanted strings
    for string in unwanted_strings:
        lyrics = lyrics.replace(string, "")

    # Remove unwanted regex
    for regex in unwanted_regex:
        lyrics = re.sub(regex, "", lyrics, flags=re.IGNORECASE)

    # Remove newlines and everything except alphanumeric strings and apostrophes
    lyrics = _remove_newlines(lyrics)
    lyrics = re.sub('[^A-Za-z0-9\s\']+', '', lyrics)

    # Check if lyrics are valid and skip them if they aren't
    lyrics_valid = _are_lyrics_valid(lyrics)
    if not lyrics_valid:
        return None

    # POS tagging
    # Get words from lyrics and POS tags from them
    words   = nltk.word_tokenize(lyrics)
    pos_arr = nltk.pos_tag(words)

    # Get all available POS tags and create dictionary with POS tags as keys
    pos_tags = app_data.POS_TAGS

    pos_tags_map = {}
    for tag in pos_tags:
        pos_tags_map[tag] = 0

    for pos in pos_arr:
        tag = pos[1]

        if pos_tags_map.get(tag) is None:
            continue

        pos_tags_map[tag] += 1

    # Normalize POS tags counts
    min_freq = sys.maxsize
    max_freq = 0
    for tag in pos_tags:
        freq = pos_tags_map[tag]
        min_freq = freq if freq < min_freq else min_freq
        max_freq = freq if freq > max_freq else max_freq

    for tag, count in pos_tags_map.items():
        normalized = (count - min_freq) / (max_freq - min_freq)
        pos_tags_map[tag] = normalized

    # Get word endings
    words = lyrics.split(" ")
    endings_dict = {}
    for word in words:
        ending = word[-3:] if len(word) > 3 else None

        if ending is None:
            continue
        else:
            endings_dict[ending] = endings_dict.get(ending, 0) + 1

    # Create dictionary with lyrics and their features
    lyrics_dict = {
        "features": {
            "verse_count"     : verse_count,                # class'd
            "stanza_count"    : stanza_count,               # class'd
            "word_count"      : total_words_count,          # class'd
            "avg_verse_length": average_verse_length,       # class'd
            "avg_verse_words" : average_verse_word_count,   # class'd
            "pos_tags_map"    : pos_tags_map,               # class'd
            "endings"         : endings_dict                # class'd
        },
        "lyrics"  : lyrics
    }

    return lyrics_dict


def _is_subgenre_allowed(genre, subgenre):

    allowed_genres = app_data.ALLOWED_SUBGENRES[genre]

    # Check if this artist's genre is allowed
    artist_genre = subgenre.replace("/", " ")
    artist_genre = artist_genre.replace(",", "")
    artist_genre_words = artist_genre.split(" ")

    for genre_word in artist_genre_words:
        if genre_word.lower() not in allowed_genres:
            return False, genre_word

    return True, None
