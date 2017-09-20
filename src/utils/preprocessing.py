import itertools
import json
import operator
import re
import sys

import nltk

import app_data as app_data
from src.utils.language_detect import LanguageDetector


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
        all_endings_dict = {}

        for index, genre in enumerate(lyrics_genres):

            file_name = lyrics_genres_file_names[index]
            file_path = Preprocessing._get_raw_lyrics_file_path(file_name)

            processed_lyrics = []

            skipped_words_map = {}
            skipped_artists_count = 0
            invalid_lyrics_count = 0

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

                        artist_genre     = artist_info["artistGenre"]
                        album_lyrics_str = artist_info["albumLyrics"]

                        allowed_check_result = Preprocessing._is_subgenre_allowed(genre, artist_genre)
                        allowed = allowed_check_result[0]

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

                                # Remove and everything except alphanumeric strings, newlines and apostrophes
                                lyrics = re.sub("[^A-Za-z0-9\\\\\s\']+", '', lyrics)

                                lyrics_valid = Preprocessing._are_lyrics_valid(lyrics)
                                if not lyrics_valid:
                                    invalid_lyrics_count += 1
                                    continue

                                processed_lyrics.append(lyrics)

                                # lyrics_dict = Preprocessing.get_lyrics_dict(lyrics)

                                # if lyrics_dict is None:
                                #     invalid_lyrics_count += 1
                                #     continue

                                # # Get word endings from current lyrics
                                # endings = lyrics_dict["features"]["endings"]
                                # for ending, count in endings.items():
                                #
                                #     # If this ending appeared for the first time, instantiate its genre dictionary
                                #     if all_endings_dict.get(ending) is None:
                                #         all_endings_dict[ending] = {}
                                #
                                #     # Inrease the count for this ending in this genre
                                #     all_endings_dict[ending][genre] = all_endings_dict[ending].get(genre, 0) + count
                                #
                                # # processed_lyrics.append(lyrics)
                                # processed_lyrics.append(lyrics_dict)

                else:
                    print("[x] \t Error: Lyrics JSON for genre " + genre + " is not in list format")

            print("Skipped artists for genre " + genre  + ": " + str(skipped_artists_count))
            print("Skipped artists per invalid word for genre " + genre + ": " + str(skipped_words_map))
            print("Skipped invalid lyrics count: " + str(invalid_lyrics_count))
            print("All endings dict size: " + str(len(all_endings_dict)))
            # Save processed lyrics for current genre
            genre_lyrics_map[genre] = processed_lyrics


        # Keep only most important endings
        # endings_with_importance = {}
        # for ending, genre_counts in all_endings_dict.items():
        #
        #     max_importance = -sys.maxsize
        #     counts = genre_counts.values()
        #     for pair in itertools.combinations(counts, 2):
        #
        #         if pair[0] < pair[1]:
        #             smaller = pair[0]
        #             bigger  = pair[1]
        #         else:
        #             smaller = pair[1]
        #             bigger  = pair[0]
        #
        #         diff = bigger - smaller
        #         importance = diff - smaller
        #
        #         if importance > max_importance:
        #             max_importance = importance
        #
        #     endings_with_importance[ending] = max_importance
        #
        # endings_with_importance_sorted_tuples = sorted(endings_with_importance.items(), key=operator.itemgetter(1), reverse=True)
        # max_importance_endings = endings_with_importance_sorted_tuples[:400]
        # max_importance_endings_dict = dict(max_importance_endings)
        #
        # del endings_with_importance
        # del endings_with_importance_sorted_tuples
        # del max_importance_endings
        #
        # # Add dictionary with all found word endings to all lyrics
        # for genre in genre_lyrics_map.keys():
        #
        #     genre_lyrics = genre_lyrics_map[genre]
        #     for index, lyrics in enumerate(genre_lyrics):
        #         endings = lyrics["features"]["endings"]
        #         new_endings = {}
        #
        #         # For every available ending:
        #         # If these lyris contain that ending, use their count
        #         # Otherwise use 0
        #         for ending in max_importance_endings_dict:
        #             new_endings[ending] = endings.get(ending, 0)
        #
        #         lyrics["features"]["endings"] = new_endings
        #         genre_lyrics[index] = lyrics
        #
        #     genre_lyrics_map[genre] = genre_lyrics

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


    @staticmethod
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
        lyrics = Preprocessing.remove_newlines(lyrics)
        lyrics = re.sub('[^A-Za-z0-9\s\']+', '', lyrics)

        # Check if lyrics are valid and skip them if they aren't
        lyrics_valid = Preprocessing._are_lyrics_valid(lyrics)
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
                "endings"         : endings_dict
            },
            "lyrics"  : lyrics
        }

        return lyrics_dict


    @staticmethod
    def _is_subgenre_allowed(genre, subgenre):

        allowed_genres = app_data.ALLOWED_SUBGENRES[genre]

        # Check if this artist's genre is allowed
        artist_genre = subgenre.replace("/", " ")
        artist_genre = artist_genre.replace(",", "")
        artist_genre_words = artist_genre.split(" ")

        for genre_word in artist_genre_words:
            if genre_word.lower() not in allowed_genres:
                return (False, genre_word)

        return (True, None)
