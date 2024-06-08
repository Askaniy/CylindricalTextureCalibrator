from typing import Callable
import FreeSimpleGUI as sg
from time import strftime

main_color = '#3884A9' # HSV 199.65° 66.86% 66.27%
#main_color = '#5D9BBA' # HSV 200.00° 50.00% 72.94% (0.5^1/2.2 = 72.974%)
text_color = '#FFFFFF'
muted_color = '#A3A3A3'
highlight_color = '#5A5A5A'
bg_color = '#333333'
inputON_color = '#424242'
inputOFF_color = '#3A3A3A'

# FreeSimpleGUI custom theme
sg.LOOK_AND_FEEL_TABLE['MaterialDark'] = {
        'BACKGROUND': bg_color, 'TEXT': text_color,
        'INPUT': inputON_color, 'TEXT_INPUT': text_color, 'SCROLL': inputON_color,
        'BUTTON': (text_color, main_color), 'PROGRESS': ('#000000', '#000000'),
        'BORDER': 0, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0
    }


def create_logger(window: sg.Window, key: str) -> Callable:
    """ Creates a function that sends messages to the window main thread """
    def logger(message: str, data=None):
        window.write_event_value((key, f'{strftime("%H:%M:%S")} {message}'), data)
    return logger

def generate_layout(img_preview_size: tuple, available_projections: tuple):
    button_size = 24
    browse_size = 10
    column1 = [
        [
            sg.Text('Select file'),
            sg.Input(enable_events=True, size=1, key='-OpenPath-', expand_x=True),
            sg.FileBrowse(size=browse_size, key='-BrowseButton-'),
        ],
        [
            sg.Text('Oblateness (1 − b/a)'),
            sg.Input('0', enable_events=True, size=1, key='-OblatenessInput-', expand_x=True),
        ],
        [sg.T('')],
        [
            sg.Checkbox('Reproject from', enable_events=True, key='-ReprojectCheckbox-'),
            sg.InputCombo(available_projections, enable_events=True, key='-ProjectionInput-'),
            sg.Text('to planetographic latitudes')
        ],
        [
            sg.Checkbox('Longitude shift (in degrees)', enable_events=True, key='-ShiftCheckbox-'),
            sg.Input('180', enable_events=True, key='-ShiftInput-', expand_x=True),
        ],
        [
            sg.Checkbox('Calibrate by color', enable_events=True, key='-ColorCheckbox-'),
            sg.Input('0.5 0.5 0.5', enable_events=True, key='-ColorInput-', expand_x=True),
        ],
        [
            sg.Checkbox('Calibrate by albedo', enable_events=True, key='-AlbedoCheckbox-'),
            sg.Input('0.5', enable_events=True, key='-AlbedoInput-', expand_x=True),
        ],
        [sg.Checkbox('Apply CIE sRGB gamma correction', enable_events=True, key='-srgbGammaCheckbox-')],
        [
            sg.Checkbox('Apply custom gamma correction', enable_events=True, key='-CustomGammaCheckbox-'),
            sg.Input('1/2.2', enable_events=True, key='-GammaInput-', expand_x=True),
        ],
        [sg.T('')],
        [
            sg.Push(),
            sg.Input(enable_events=True, key='-SaveFolder-', visible=False),
            sg.FolderBrowse('Process', size=button_size, key='-ProcessButton-'),
            sg.Push(),
        ],
    ]

    column2 = [
        [sg.Push(), sg.Text('Input preview', key='-InputTitle-'), sg.Push()],
        [sg.Push(), sg.Image(background_color='black', size=img_preview_size, key='-InputPreview-'), sg.Push()],
        [sg.T('')],
        [sg.Push(), sg.Text('Output preview', key='-OutputTitle-'), sg.Push()],
        [sg.Push(), sg.Image(background_color='black', size=img_preview_size, key='-OutputPreview-'), sg.Push()],
    ]

    return [
        [
            sg.Column(column1),
            #sg.VSeperator(),
            sg.Column(column2, expand_x=True),
        ]
    ]