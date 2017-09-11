
# Project folder paths
RESOURCES_PATH = "../res"

# Lyrics folder paths
LYRICS_PATH                      = RESOURCES_PATH + "/lyrics"
RAW_LYRICS_FOLDER_PATH           = LYRICS_PATH + "/raw"
FILTERED_LYRICS_FOLDER_PATH      = LYRICS_PATH + "/filtered"
PREPROCESSED_LYRICS_FOLDER_PATH  = LYRICS_PATH + "/preprocessed"
LYRICS_WITH_FEATURES_FOLDER_PATH = LYRICS_PATH + "/featurized"

# Lyrics file paths
FILTERED_LYRICS_FILE_PATH      = FILTERED_LYRICS_FOLDER_PATH + "/filtered_lyrics.txt"
FILTERED_LYRICS_FILE_PATH_TEST = FILTERED_LYRICS_FOLDER_PATH + "/filtered_lyrics_test.text"

PREPROCESSED_LYRICS_FILE_PATH      = PREPROCESSED_LYRICS_FOLDER_PATH + "/preprocessed.txt"
PREPROCESSED_LYRICS_FILE_PATH_TEST = PREPROCESSED_LYRICS_FOLDER_PATH + "/preprocessed_test.txt"

LYRICS_WITH_FEATURES_FILE_PATH      = LYRICS_WITH_FEATURES_FOLDER_PATH + "/featurized.txt"
LYRICS_WITH_FEATURES_FILE_PATH_TEST = LYRICS_WITH_FEATURES_FOLDER_PATH + "/featurized_test.txt"

# Pickle paths
MODEL_FOLDER_PATH = RESOURCES_PATH + "/model"
PICKLE_FILE_PATH  = MODEL_FOLDER_PATH + "/pickle.pkl"

# Application variables
LYRICS_GENRES = ["black", "death", "doom", "thrash"]
BATCH_SIZE    = 80

PREPROCESS_LYRICS_UNWANTED_STRINGS = ["webmaster@darklyrics.com", "submits, comments, corrections are welcomed at", "lyrics"]
PREPROCESS_LYRICS_UNWANTED_REGEX   = ["[[].*[]]", "<a>.*<\/a>", "<(?:.|\n)*?>", "\d.", "(thanks).*(lyrics)", "(Thanks to).*(lyrics)"] ###"(All lyrics).*", "(Lyrics by).*", "(Produced by).*", "(Recorded & mixed).*", , "(Lyrics written).*"]

MIN_LYRICS_LENGTH = 80
LYRICS_SEPARATOR  = "|$%:%$|"

ALLOWED_SUBGENRES = {
    "death"  : ["death", "metal", "brutal", "technical", "tech", "progressive"],
    "thrash" : ["thrash", "metal", "speed"],
    "doom"   : ["doom", "metal", "funeral", "atmospheric", "dark", "ambient", "drone"],
    "black"  : ["black", "metal", "pagan"]
}

POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

VERSE_BREAK  = "<br>\\n"
STANZA_BREAK = "<br>\\n<br>\\n"


# Boolean properties to control application actions
DEBUG_MODE        = False
FILTER_LYRICS     = False
PREPROCESS_LYRICS = True
TRAIN_CLASSIFIER  = True

CC_SAMPLE = "Early hours open road family of five on their way home Having enjoyed a day in the sun their encounter with gore has just begun A homicidal fool not knowing left from right now has the family in his sight Trying to perceive if he's blind or insane he steers his car into the other lane Both of them collide expressions horrified Head on at full speed the vultures will soon feed The father of three was impaled on the wheel as his skull became a part of the dash His eyeballs ejected his sight uneffected he saw his own organs collapse His seatbelt was useless for holding him back it simply cut him in two Legs were crushed out leaked pus as his spinal cord took off and flew The mother took flight through the glass and ended up impaled on a sign Her intestines stretched from the car down the road for a quarter of a mile Fourth child on the way won't live another day Fetus on the road with mangled little bones Little children fly not a chance to wonder why Smashed against the ceiling all their skin burning and peeling Shards of glass explode chest and skull now implode Corpses they've become and graves will have to be dug Underneath the wheels burning rubber on your face Bleeding from your eyes the slaughtered victims lies Knowing what he's done he just backs up one more time Laughing at the mess a pile of meat on the street One child left slowly dying now arteries gushing blood Now it's time to feed on flesh the gore has just begun  Early hours open road family of five  on their way home Having enjoyed a day in the sun their encounter with gore has just begun A homicidal fool not knowing left from right now has the family in his sight Trying to perceive if he's blind or insane he steers his car into the other lane The look of death in my eye Surely noone will survive Just a pile of mush Left to dry in the sun I see my fresh kill Left in the road Remains of your bodies Mangled and torn"

SB_SAMPLE = "Remember my love the object we saw That beautiful morning in June: By a bend in the path a carcass reclined On a bed sown with pebbles and stones; Her legs were spread out like a lecherous whore Sweating out poisonous fumes Who opened in slick invitational style Her stinking and festering womb The sun on this rottenness focused its rays To cook the cadaver till done And render to Nature a hundredfold gift Of all she'd united in one And the sky cast an eye on this marvellous meat As over the flowers in bloom The stench was so wretched that there on the grass You nearly collapsed in a swoon The flies buzzed and droned on these bowels of filth Where an army of maggots arose Which flowed with a liquid and thickening stream On the animate rags of her clothes And it rose and it fell and pulsed like a wave Rushing and bubbling with health One could say that this carcass blown with vague breath Lived in increasing itself And this whole teeming world made a musical sound Like babbling brooks and the breeze Or the grain that a man with a winnowingfan Turns with a rhythmical ease The shapes wore away as if only a dream Like a sketch that is left on the page Which the artist forgot and can only complete On the canvas with memory's aid From back in the rocks a pitiful bitch Eyed us with angry distaste Awaiting the moment to snatch from the bones The morsel she'd dropped in her haste And you in your turn will be rotten as this: Horrible filthy undone O sun of my nature and star of my eyes My passion my angel in one Yes such will you be o regent of grace After the rites have been read Under the weeds under blossoming grass As you moulder with bones of the dead Ah then o my beauty explain to the worms Who cherish your body so tine That I am the keeper for corpses of love Of the form and the essence divine"