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
text_colors = (text_color, muted_color)

# FreeSimpleGUI custom theme
sg.LOOK_AND_FEEL_TABLE['MaterialDark'] = {
    'BACKGROUND': bg_color, 'TEXT': text_color,
    'INPUT': inputON_color, 'TEXT_INPUT': text_color, 'SCROLL': inputON_color,
    'BUTTON': (text_color, main_color), 'PROGRESS': ('#000000', '#000000'),
    'BORDER': 0, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0
}


planetographic_latitudes_texts = ('', '(converted to planetographic latitudes)')

def create_logger(window: sg.Window, key: str) -> Callable:
    """ Creates a function that sends messages to the window main thread """
    def logger(message: str, data=None):
        window.write_event_value((key, f'{strftime("%H:%M:%S")} {message}'), data)
    return logger

def generate_layout(
        img_preview_size: tuple[int, int],
        is_gamma_corrected: bool,
        latitude_systems: tuple[str, ...],
        default_latitude_system: str,
        is_cylindrical_map: bool,
        is_latitude_converted: bool,
        oblateness: int|float
    ):
    button_size = 24
    browse_size = 10
    column0 = [
        [sg.Push(), sg.Text('Input preview', key='-InputTitle-'), sg.Push()],
        [
            #sg.Text('Select file'),
            sg.Input(enable_events=True, size=1, key='-OpenFileName-', expand_x=True),
            sg.FileBrowse(size=browse_size, initial_folder='..', key='-BrowseButton-'),
        ],
        [sg.Push(), sg.Image(background_color=None, size=img_preview_size, key='-InputPreview-'), sg.Push()],
        [
            sg.Push(),
            sg.Input(
                'Linear input brightness will be here', size=36,
                readonly=True, disabled_readonly_background_color=inputOFF_color,
                key='-InputRGB-', expand_x=True
            ),
            sg.Push()
        ],
        [sg.T('')],
        [sg.Checkbox('Is gamma corrected (CIE sRGB)', default=is_gamma_corrected, enable_events=True, key='-IsGammaCorrected-')],
        [sg.Checkbox('Is a cylindrical map', default=is_cylindrical_map, enable_events=True, key='-IsCylindricalMap-')],
        [
            sg.Text('Latitude system:', text_color=text_colors[not is_cylindrical_map], key='-LatitudeSystemText-'),
            sg.Combo(latitude_systems, default_value=default_latitude_system, enable_events=True, key='-LatitudeSystemInput-'),
        ],
        [
            sg.Text('Oblateness (1 − b/a)', text_color=text_colors[not is_latitude_converted], key='-OblatenessText-'),
            sg.Input(
                str(oblateness), enable_events=True, size=12,
                disabled=not is_latitude_converted, disabled_readonly_background_color=inputOFF_color,
                disabled_readonly_text_color=muted_color, key='-OblatenessInput-', expand_x=True
            ),
        ],
    ]
    column1 = [
        [sg.Push(), sg.Text('Processing', key='-ProcessingTitle-'), sg.Push()],
        [sg.T('')],
        [
            sg.Checkbox('Longitude shift (in degrees)', enable_events=True, key='-ShiftCheckbox-'),
            sg.Input('180', size=12, enable_events=True, key='-ShiftInput-', expand_x=True),
        ],
        [
            sg.Checkbox('Calibrate by linear color', enable_events=True, key='-ColorCheckbox-'),
            sg.Input('0.5 0.5 0.5', size=12, enable_events=True, key='-ColorInput-', expand_x=True),
        ],
        [
            sg.Checkbox('Calibrate by albedo', enable_events=True, key='-AlbedoCheckbox-'),
            sg.Input('0.5', size=12, enable_events=True, key='-AlbedoInput-', expand_x=True),
        ],
        [sg.Checkbox('Overlay a coordinate grid', enable_events=True, key='-GridCheckbox-')],
        [sg.T('')],
        [sg.Checkbox('Apply CIE sRGB gamma correction', enable_events=True, key='-srgbGammaCheckbox-')],
        #[
        #    sg.Checkbox('Apply custom gamma correction', enable_events=True, key='-CustomGammaCheckbox-'),
        #    sg.Input('1/2.2', size=12, enable_events=True, key='-GammaInput-', expand_x=True),
        #],
        [sg.Checkbox('Maximize brightness', enable_events=True, key='-MaximizeBrightnessCheckbox-')],
    ]

    column2 = [
        [sg.Push(), sg.Text('Output preview', key='-OutputTitle-'), sg.Push()],
        [
            sg.Push(),
            sg.Text(planetographic_latitudes_texts[is_latitude_converted], key='-PlanetographicLatitudesText-'),
            sg.Push()
        ],
        [sg.Push(), sg.Image(background_color=None, size=img_preview_size, key='-OutputPreview-'), sg.Push()],
        [
            sg.Push(),
            sg.Input(
                'Final brightness will be here', size=36,
                readonly=True, disabled_readonly_background_color=inputOFF_color,
                key='-OutputRGB-', expand_x=True
            ),
            sg.Push()
        ],
        [sg.T('')],
        [
            sg.Push(),
            sg.Input(enable_events=True, key='-SaveFileName-', visible=False),
            sg.FileSaveAs('Process and save', size=button_size, key='-ProcessButton-'),
            sg.Push(),
        ],
    ]

    return [
        [
            sg.Column(column0, vertical_alignment='top'),
            sg.VSeperator(),
            sg.Column(column1, vertical_alignment='top'),
            sg.VSeperator(),
            sg.Column(column2, vertical_alignment='top'),
        ]
    ]
