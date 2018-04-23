import langdetect

class LanguageDetector(object):

    @staticmethod
    def is_correct_language(text, test_lang):
        """
        Checks if the given text is written in the given language.

        :param text: str
        :param test_lang: str
        :return: boolean
        """
        detected_lang = langdetect.detect(text)
        return detected_lang == test_lang

    @staticmethod
    def is_english(text):
        """
        Checks if the given text is written in english.

        :param text: The text to check
        :return: Boolean value, true if text is in english, false otherwise
        """
        try:
            detected_lang = langdetect.detect(text)
            return detected_lang == 'en'
        except:
            return False
