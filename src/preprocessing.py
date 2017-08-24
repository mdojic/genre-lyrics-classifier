import src.app_data as app_data
import re

from src.lyrics_reader import LyricsReader

class Preprocessing(object):

    @staticmethod
    def filter_lyrics():

        # Get available genres
        lyrics_genres = app_data.LYRICS_GENRES

        # Check if app is in debug mode
        is_debug_mode = app_data.DEBUG_MODE

        # If app is in debug mode, use test files
        if is_debug_mode:
            lyrics_genres_file_names = [genre + "_test.txt" for genre in lyrics_genres]
        else:
            lyrics_genres_file_names = [genre + ".txt" for genre in lyrics_genres]

        # Read lyrics from all input files
        genre_lyrics_map = {}
        for index, genre in enumerate(lyrics_genres):
            file_name = lyrics_genres_file_names[index]
            file_path = Preprocessing._get_raw_lyrics_file_path(file_name)

            lyrics = LyricsReader.read_english_lyrics_from_file(file_path)
            lyrics = Preprocessing._preprocess_lyrics(lyrics)
            genre_lyrics_map[genre] = lyrics

        return genre_lyrics_map


    @staticmethod
    def _preprocess_lyrics(lyrics):
        unwanted_strings = app_data.PREPROCESS_LYRICS_UNWANTED_STRINGS
        unwanted_regex = app_data.PREPROCESS_LYRICS_UNWANTED_REGEX

        for index, song_lyrics in enumerate(lyrics):

            # Remove unwanted strings from lyrics
            for single_unwanted_string in unwanted_strings:
                song_lyrics = song_lyrics.replace(single_unwanted_string, "")

            # Remove unwanted patterns by regex from lyrics
            for single_unwanted_regex in unwanted_regex:
                song_lyrics = re.sub(single_unwanted_regex, "", song_lyrics, flags=re.IGNORECASE)

            lyrics[index] = song_lyrics

        return lyrics


    @staticmethod
    def _get_raw_lyrics_file_path(file_name):
        return app_data.RAW_LYRICS_FOLDER_PATH + "/" + file_name