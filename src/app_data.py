
# Project folder paths
RESOURCES_PATH                 = "../res"
LYRICS_PATH                    = RESOURCES_PATH + "/lyrics"
RAW_LYRICS_FOLDER_PATH         = LYRICS_PATH + "/raw"
FILTERED_LYRICS_FOLDER_PATH    = LYRICS_PATH + "/filtered"
FILTERED_LYRICS_FILE_PATH      = FILTERED_LYRICS_FOLDER_PATH + "/filtered_lyrics.txt"
FILTERED_LYRICS_FILE_PATH_TEST = FILTERED_LYRICS_FOLDER_PATH + "/filtered_lyrics_test.text"
MODEL_FOLDER_PATH              = RESOURCES_PATH + "/model"
PICKLE_FILE_PATH               = MODEL_FOLDER_PATH + "/pickle.pkl"

# Application variables
LYRICS_GENRES          = ["black", "death", "doom", "thrash"]
BATCH_SIZE = 250
PREPROCESS_LYRICS_UNWANTED_STRINGS = ["webmaster@darklyrics.com", "Submits, comments, corrections are welcomed at"]
PREPROCESS_LYRICS_UNWANTED_REGEX = ["[[].*[]]", "(thanks).*(lyrics)", "(All lyrics).*", "(Lyrics by).*", "(Produced by).*", "(Recorded & mixed).*", "(Thanks to).*(lyrics)", "(Lyrics written).*"]

# Boolean properties to control application actions
DEBUG_MODE    = False
FILTER_LYRICS = False