import os

# Project folder paths
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES_PATH = APP_ROOT + "/res"

# Lyrics folder paths
LYRICS_PATH                      = RESOURCES_PATH + "/lyrics"
RAW_LYRICS_FOLDER_PATH           = LYRICS_PATH + "/raw"
PREPROCESSED_LYRICS_FOLDER_PATH  = LYRICS_PATH + "/preprocessed"
DATASET_LYRICS_FOLDER_PATH       = LYRICS_PATH + "/dataset"

# Subfolders for raw lyrics
RAW_LYRICS_SUBFOLDER_PATH_METAL  = RAW_LYRICS_FOLDER_PATH + "/metal"
RAW_LYRICS_SUBFOLDER_PATH_OTHER  = RAW_LYRICS_FOLDER_PATH + "/other"

PREPROCESSED_LYRICS_SUBFOLDER_PATH_METAL = PREPROCESSED_LYRICS_FOLDER_PATH + "/metal"
PREPROCESSED_LYRICS_SUBFOLDER_PATH_ALL   = PREPROCESSED_LYRICS_FOLDER_PATH + "/all_genres"

# Preprocessed lyrics paths
PREPROCESSED_LYRICS_FILE_PATH_METAL      = PREPROCESSED_LYRICS_SUBFOLDER_PATH_METAL + "/preprocessed.txt"
PREPROCESSED_LYRICS_FILE_PATH_METAL_TEST = PREPROCESSED_LYRICS_SUBFOLDER_PATH_METAL + "/preprocessed_test.txt"

PREPROCESSED_LYRICS_FILE_PATH_ALL      = PREPROCESSED_LYRICS_SUBFOLDER_PATH_ALL + "/preprocessed.txt"
PREPROCESSED_LYRICS_FILE_PATH_ALL_TEST = PREPROCESSED_LYRICS_SUBFOLDER_PATH_ALL + "/preprocessed_test.txt"

# Dataset lyrics subfolders
TRAINING_DATASET_SUBFOLDER_PATH_METAL = DATASET_LYRICS_FOLDER_PATH + "/metal"
TRAINING_DATASET_SUBFOLDER_PATH_ALL   = DATASET_LYRICS_FOLDER_PATH + "/all_genres"

# Dataset lyrics paths
TRAINING_DATASET_LYRICS_FILE_PATH_METAL = TRAINING_DATASET_SUBFOLDER_PATH_METAL + "/training.txt"
TEST_DATASET_LYRICS_FILE_PATH_METAL     = TRAINING_DATASET_SUBFOLDER_PATH_METAL + "/test.txt"

TRAINING_DATASET_LYRICS_FILE_PATH_ALL = TRAINING_DATASET_SUBFOLDER_PATH_ALL + "/training.txt"
TEST_DATASET_LYRICS_FILE_PATH_ALL     = TRAINING_DATASET_SUBFOLDER_PATH_ALL + "/test.txt"

# Pickle paths
MODEL_FOLDER_PATH = RESOURCES_PATH + "/model"

PICKLE_FILE_PATH_METAL = MODEL_FOLDER_PATH + "/pickle_metal.pkl"
PICKLE_FILE_PATH_ALL   = MODEL_FOLDER_PATH + "/pickle_all.pkl"

# Application variables
LYRICS_GENRES_METAL = ["black", "death", "doom", "thrash"]
LYRICS_GENRES_OTHER = ["pop", "rap"]
LYRICS_GENRES_ALL   = ["pop", "rap", "metal"]
BATCH_SIZE    = 80

PREPROCESS_LYRICS_UNWANTED_STRINGS = ["webmaster@darklyrics.com", "submits, comments, corrections are welcomed at", "lyrics"]
PREPROCESS_LYRICS_UNWANTED_REGEX   = ["[[].*[]]", "<a>.*<\/a>", "<(?:.|\n)*?>", "\d.",
                                      "(thanks).*(lyrics)", "(Thanks to).*(lyrics)"]

MIN_LYRICS_LENGTH = 80
LYRICS_SEPARATOR  = "|$%:%$|"

ALLOWED_SUBGENRES = {
    "death"  : ["death", "metal", "brutal", "technical", "tech", "progressive"],
    "thrash" : ["thrash", "metal", "speed"],
    "doom"   : ["doom", "metal", "funeral", "atmospheric", "dark", "ambient", "drone"],
    "black"  : ["black", "metal", "pagan"]
}

POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
TOP_ENDINGS_COUNT = 400

VERSE_BREAK  = "<br>\\n"
STANZA_BREAK = "<br>\\n<br>\\n"

LYRICS_VERSE_BREAK  = "\\n"
LYRICS_STANZA_BREAK = "\\n"

# =================================================================
#  Boolean properties to control application actions
DEBUG_MODE         = False      # Whether to run the app in debug mode (use test files with less data etc)
PREPROCESS_LYRICS  = False      # Whether to preprocess lyrics when the app starts
TRAIN_CLASSIFIER   = True       # Whether to train the classifier and pickle it, or to load a pickled one
SPLIT_DATASET      = True      # Whether to split lyrics datasets into training and test set files
METAL_ONLY         = False       # Whether to run the app for metal subgenres, or for "big" genres
LIMIT_LYRICS_COUNT = True       # Whether to limit the max amount of lyrics per genre or use all available lyrics
# =================================================================

# Cannibal corpse lyrics sample (death metal)
CC_SAMPLE = "Early hours open road family of five on their way home Having enjoyed a day in the sun their encounter with gore has just begun A homicidal fool not knowing left from right now has the family in his sight Trying to perceive if he's blind or insane he steers his car into the other lane Both of them collide expressions horrified Head on at full speed the vultures will soon feed The father of three was impaled on the wheel as his skull became a part of the dash His eyeballs ejected his sight uneffected he saw his own organs collapse His seatbelt was useless for holding him back it simply cut him in two Legs were crushed out leaked pus as his spinal cord took off and flew The mother took flight through the glass and ended up impaled on a sign Her intestines stretched from the car down the road for a quarter of a mile Fourth child on the way won't live another day Fetus on the road with mangled little bones Little children fly not a chance to wonder why Smashed against the ceiling all their skin burning and peeling Shards of glass explode chest and skull now implode Corpses they've become and graves will have to be dug Underneath the wheels burning rubber on your face Bleeding from your eyes the slaughtered victims lies Knowing what he's done he just backs up one more time Laughing at the mess a pile of meat on the street One child left slowly dying now arteries gushing blood Now it's time to feed on flesh the gore has just begun  Early hours open road family of five  on their way home Having enjoyed a day in the sun their encounter with gore has just begun A homicidal fool not knowing left from right now has the family in his sight Trying to perceive if he's blind or insane he steers his car into the other lane The look of death in my eye Surely noone will survive Just a pile of mush Left to dry in the sun I see my fresh kill Left in the road Remains of your bodies Mangled and torn"

# Saban Saulic lyrics sample (serbian folk)
SB_SAMPLE = "Remember my love the object we saw That beautiful morning in June: By a bend in the path a carcass reclined On a bed sown with pebbles and stones; Her legs were spread out like a lecherous whore Sweating out poisonous fumes Who opened in slick invitational style Her stinking and festering womb The sun on this rottenness focused its rays To cook the cadaver till done And render to Nature a hundredfold gift Of all she'd united in one And the sky cast an eye on this marvellous meat As over the flowers in bloom The stench was so wretched that there on the grass You nearly collapsed in a swoon The flies buzzed and droned on these bowels of filth Where an army of maggots arose Which flowed with a liquid and thickening stream On the animate rags of her clothes And it rose and it fell and pulsed like a wave Rushing and bubbling with health One could say that this carcass blown with vague breath Lived in increasing itself And this whole teeming world made a musical sound Like babbling brooks and the breeze Or the grain that a man with a winnowingfan Turns with a rhythmical ease The shapes wore away as if only a dream Like a sketch that is left on the page Which the artist forgot and can only complete On the canvas with memory's aid From back in the rocks a pitiful bitch Eyed us with angry distaste Awaiting the moment to snatch from the bones The morsel she'd dropped in her haste And you in your turn will be rotten as this: Horrible filthy undone O sun of my nature and star of my eyes My passion my angel in one Yes such will you be o regent of grace After the rites have been read Under the weeds under blossoming grass As you moulder with bones of the dead Ah then o my beauty explain to the worms Who cherish your body so tine That I am the keeper for corpses of love Of the form and the essence divine"


def get_preprocessed_lyrics_path():

    metal_path = PREPROCESSED_LYRICS_FILE_PATH_METAL
    all_path   = PREPROCESSED_LYRICS_FILE_PATH_ALL

    metal_path_test = PREPROCESSED_LYRICS_FILE_PATH_METAL_TEST
    all_path_test   = PREPROCESSED_LYRICS_FILE_PATH_ALL_TEST

    if DEBUG_MODE:
        return metal_path_test if METAL_ONLY else all_path_test
    else:
        return metal_path if METAL_ONLY else all_path


def get_training_set_file_path():
    return TRAINING_DATASET_LYRICS_FILE_PATH_METAL if METAL_ONLY else TRAINING_DATASET_LYRICS_FILE_PATH_ALL


def get_test_set_file_path():
    return TEST_DATASET_LYRICS_FILE_PATH_METAL if METAL_ONLY else TEST_DATASET_LYRICS_FILE_PATH_ALL


def get_classifier_pickle_path():
    return PICKLE_FILE_PATH_METAL if METAL_ONLY else PICKLE_FILE_PATH_ALL


def get_genres():
    return LYRICS_GENRES_METAL if METAL_ONLY else LYRICS_GENRES_ALL

