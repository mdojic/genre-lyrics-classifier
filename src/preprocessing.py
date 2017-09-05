import src.app_data as app_data

import re
import json
import nltk
import sys

from src.lyrics_reader import LyricsReader
from src.language_detect import LanguageDetector


class Preprocessing(object):

    @staticmethod
    def preprocess_lyrics():

        # Get available genres
        lyrics_genres = app_data.LYRICS_GENRES

        # If app is in debug mode, use test files
        if app_data.DEBUG_MODE:
            lyrics_genres_file_names = [genre + "_test.txt" for genre in lyrics_genres]
        else:
            lyrics_genres_file_names = [genre + ".txt" for genre in lyrics_genres]

        genre_lyrics_map = {}

        for index, genre in enumerate(lyrics_genres):

            file_name = lyrics_genres_file_names[index]
            file_path = Preprocessing._get_raw_lyrics_file_path(file_name)

            processed_lyrics = []

            allowed_genres = app_data.ALLOWED_SUBGENRES[genre]

            skipped_words_map = {}
            skipped_artists_count = 0
            invalid_lyrics_count = 0

            with open(file_path, "rt", encoding="utf-8") as file:

                # Get content from file
                content = json.loads(file.read())

                # Strings and regex for filtering lyrics
                unwanted_strings = app_data.PREPROCESS_LYRICS_UNWANTED_STRINGS
                unwanted_regex   = app_data.PREPROCESS_LYRICS_UNWANTED_REGEX

                print("Got JSON from file " + file_name)

                # Check if file content is a JSON list
                json_is_list = type(content) is list

                if json_is_list:

                    print("[✔] \t JSON is in list format")

                    # Go through list of artist JSONs and process them
                    for artist_info in content:

                        artist_name      = artist_info["artistName"]
                        artist_genre     = artist_info["artistGenre"]
                        album_lyrics_str = artist_info["albumLyrics"]

                        # Check if this artist's genre is allowed
                        artist_genre = artist_genre.replace("/", " ")
                        artist_genre = artist_genre.replace(",", "")
                        artist_genre_words = artist_genre.split(" ")

                        allowed = True
                        for genre_word in artist_genre_words:

                            # Skip this artist if his genre is not allowed
                            if genre_word.lower() not in allowed_genres:
                                allowed = False
                                skipped_words_map[genre_word] = skipped_words_map[genre_word] + 1 if skipped_words_map.get(genre_word) != None else 1
                                break

                        if not allowed:
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

                                verse_break  = app_data.VERSE_BREAK
                                stanza_break = app_data.STANZA_BREAK

                                # Count verses and stanzas in song lyrics (add one because breaks are only between verses/stanzas)
                                verse_count  = lyrics.count(verse_break)  + 1
                                stanza_count = lyrics.count(stanza_break) + 1

                                # Extract average verse length feature
                                verses = lyrics.split(verse_break)
                                number_of_verses = len(verses)

                                total_verses_length = 0
                                for verse in verses:
                                    verse_length = len(verse)
                                    total_verses_length += verse_length

                                average_verse_length = total_verses_length / number_of_verses

                                # Remove unwanted strings
                                for string in unwanted_strings:
                                    lyrics = lyrics.replace(string, "")

                                # Remove unwanted regex
                                for regex in unwanted_regex:
                                    lyrics = re.sub(regex, "", lyrics, flags=re.IGNORECASE)


                                # Remove newlines and everything except alphanumeric strings and apostrophes
                                lyrics = Preprocessing.remove_newlines(lyrics)
                                lyrics = re.sub('[^A-Za-z0-9\s\']+', '', lyrics)

                                # Check if lyrics are valid and skip them if they aren't
                                lyrics_valid = Preprocessing._are_lyrics_valid(lyrics)
                                if not lyrics_valid:
                                    invalid_lyrics_count += 1
                                    continue

                                # POS tagging
                                # Get words from lyrics and POS tags from them
                                words = nltk.word_tokenize(lyrics)
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

                                    pos_tags_map[tag] = pos_tags_map[tag] + 1
                                    pass

                                min_freq = sys.maxsize
                                max_freq = 0
                                for tag in pos_tags:
                                    freq = pos_tags_map[tag]
                                    min_freq = freq if freq < min_freq else min_freq
                                    max_freq = freq if freq > max_freq else max_freq

                                for tag, count in pos_tags_map.items():
                                    normalized = (count - min_freq) / (max_freq - min_freq)
                                    pos_tags_map[tag] = normalized

                                # Create dictionary with lyrics and their features
                                lyrics_json = {
                                    "features" : {
                                        "verse_count"      : verse_count,
                                        "stanza_count"     : stanza_count,
                                        "avg_verse_length" : average_verse_length,
                                        "pos_tags_map"     : pos_tags_map
                                    },
                                    "lyrics" : lyrics
                                }

                                # processed_lyrics.append(lyrics)
                                processed_lyrics.append(lyrics_json)

                else:
                    print("[x] \t Error: Lyrics JSON for genre " + genre + " is not in list format")

            print("Skipped artists for genre: " + str(skipped_artists_count))
            print("Skipped artists per invalid word for genre " + genre + ": " + str(skipped_words_map))
            print("Skipped invalid lyrics count: " + str(invalid_lyrics_count))

            # Save processed lyrics for current genre
            genre_lyrics_map[genre] = processed_lyrics


        Preprocessing._save_preprocessed_lyrics(genre_lyrics_map)


    @staticmethod
    def remove_newlines(string):
        result = string.replace("\\n", " ")
        return result


    @staticmethod
    def _get_raw_lyrics_file_path(file_name):
        return app_data.RAW_LYRICS_FOLDER_PATH + "/" + file_name


    @staticmethod
    def _save_preprocessed_lyrics(genre_lyrics_map):

        if app_data.DEBUG_MODE:
            output_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH_TEST
        else:
            output_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH

        with open(output_file_path, 'w') as output_file:
            json.dump(genre_lyrics_map, output_file)


    @staticmethod
    def _are_lyrics_valid(lyrics):

        min_length = app_data.MIN_LYRICS_LENGTH

        lyrics_are_english = LanguageDetector.is_english(lyrics)
        lyrics_length_ok   = len(lyrics) >= min_length

        return lyrics_are_english and lyrics_length_ok