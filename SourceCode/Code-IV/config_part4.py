import os

OUTPUT_FOLDER = 'Report/OUTPUT/OUTPUT_PART4'
PART1_RESULTS_FILE = os.path.join('Report/OUTPUT/OUTPUT_PART1', 'results.csv')
RAW_DATA_FILENAME = os.path.join(OUTPUT_FOLDER, 'raw_data_with_highest_etv.csv')
ESTIMATION_READY_DATA_FILENAME = os.path.join(OUTPUT_FOLDER, 'estimation_data_with_highest_etv.csv')

TRANSFER_URL = 'https://www.footballtransfers.com/us/players/uk-premier-league'

PLAYER_TABLE_SELECTOR = 'tbody#player-table-body'
PLAYER_ROW_SELECTOR = 'tr'
PLAYER_NAME_SELECTOR = 'td.td-player div.text > a'
TEAM_NAME_SELECTOR = 'td.td-team span.td-team__teamname'
ETV_SELECTOR = 'td.text-center > span.player-tag'
AGE_SELECTOR = 'td.m-hide.age'
POSITION_SELECTOR = 'td.td-player span.sub-text.d-none.d-md-block'
NEXT_PAGE_SELECTOR = 'button.pagination_next_button:not([disabled])'

HIGHEST_ETV_SELECTOR_PROFILE = 'div.player-key span.player-tag:not(.player-tag-dark)'
PROFILE_PAGE_LOAD_CHECK_SELECTOR = 'div.playerInfoTop-bar'

WAIT_TIME = 15 # seconds

MIN_MINUTES_THRESHOLD = 900
PART1_MINUTES_COLUMN = 'Playing Time: minutes'
PART1_PLAYER_COLUMN = 'Player'
TARGET_VARIABLE = 'TransferValue_EUR_Millions'