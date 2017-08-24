from src.language_detect import LanguageDetector


class LyricsReader(object):

    LYRICS_SEPARATOR = "|$%:%$|"
    MIN_LYRICS_LENGTH = 80

    @staticmethod
    def read_english_lyrics_from_file_streaming(file_path):

        """
        Read lyrics from a file and return an array containing only lyrics in english.

        :param file_path: path to the file containing the lyrics
        :return: array of strings containing all english lyrics from the file
        """

        current_lyrics = ""
        lyrics_list    = []

        separator            = LyricsReader.LYRICS_SEPARATOR
        separator_chars_read = 0
        separator_length     = len(separator)

        min_length = LyricsReader.MIN_LYRICS_LENGTH

        file = open(file_path, "rt", encoding="utf-8")

        next = file.read(1)
        while next != "":

            if separator_chars_read == separator_length:

                separator_chars_read = 0

                # Consider only lyrics which are in english and long enough
                lyrics_are_english = LanguageDetector.is_english(current_lyrics)
                lyrics_length_ok   = len(current_lyrics) >= min_length
                if (lyrics_are_english and lyrics_length_ok):
                    lyrics_list.append(current_lyrics)

                current_lyrics = ""

            elif next == separator[separator_chars_read]:

                separator_chars_read = separator_chars_read + 1

            else:
                current_lyrics += next
                separator_chars_read = 0

            next = file.read(1)

        return lyrics_list


    @staticmethod
    def read_english_lyrics_from_file(file_path):

        """

        Read english lyrics from the given file

        :param file_path: Path to the file containing the lyrics to be read
        :return: Array containing only english lyrics from the input file
        """

        print("Reading lyrics from file: " + file_path)

        # Get raw lyrics separator and minimum lyrics length
        separator  = LyricsReader.LYRICS_SEPARATOR
        min_length = LyricsReader.MIN_LYRICS_LENGTH

        # Open file with lyrics
        file = open(file_path, "rt", encoding="utf-8")

        # Get lyrics arrays by splitting file content on separator
        content      = file.read()
        lyrics_array = content.split(separator)
        result       = []

        # Keep only lyrics which are in english and long enough
        for current_lyrics in lyrics_array:

            lyrics_are_english = LanguageDetector.is_english(current_lyrics)
            lyrics_length_ok   = len(current_lyrics) >= min_length

            if (lyrics_are_english and lyrics_length_ok):
                result.append(current_lyrics)

        return result

