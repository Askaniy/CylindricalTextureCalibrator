import FreeSimpleGUI as sg

import src.gui as gui
import src.auxiliary as aux
from PIL import Image
import numpy as np


def launch_window():

    # GUI configuration
    preview_size = (256, 128)
    preview_area = preview_size[0]*preview_size[1]
    is_cylindrical_map = True
    is_latitude_converted = False
    latitude_systems = tuple(aux.latitude_systems_dict.keys())
    oblateness = 0

    # Loading the icon
    with open('src/images/icon', 'rb') as file:
        icon = file.read()

    # Launching window
    sg.theme('MaterialDark')
    window = sg.Window(
        'Cylindrical Texture Calibrator', icon=icon, finalize=True, resizable=True,
        layout=gui.generate_layout(
            preview_size, latitude_systems, is_cylindrical_map, is_latitude_converted, oblateness
        )
    )

    # GUI setup
    logger = gui.create_logger(window, '-Thread-')
    image = preview = None

    # Creating the default working object
    black_rectangle = np.zeros(preview_size, dtype='uint8')
    img_arr = aux.ImageArray(Image.fromarray(black_rectangle), is_cylindrical_map, oblateness)

    mean_brightness_triggers = (
        '-OpenFileName-', '-IsGammaCorrected-', '-IsCylindricalMap-',
        '-LatitudeSystemInput-', '-OblatenessInput-'
    )

    # Window events loop
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        # The checkbox causes updates for GUI elements
        # LatitudeSystem is used only if the input is a cylindrical map
        # Oblateness is used only if the latitude system needs convertion to planetographic
        elif event in ('-IsCylindricalMap-', '-LatitudeSystemInput-'):
            is_cylindrical_map = values['-IsCylindricalMap-']
            is_latitude_converted = is_cylindrical_map and (values['-LatitudeSystemInput-'] is not latitude_systems[0])
            window['-LatitudeSystemText-'].update(text_color=gui.text_colors[not is_cylindrical_map])
            window['-OblatenessText-'].update(text_color=gui.text_colors[not is_latitude_converted])
            window['-OblatenessInput-'].update(disabled=not is_latitude_converted)
            window['-PlanetographicLatitudesText-'].update(gui.planetographic_latitudes_texts[is_latitude_converted])
            img_arr.is_cylindrical_map = is_cylindrical_map

        # Opening a file starts processing thread
        elif event == '-OpenFileName-':
            image, preview = aux.image_reader(values['-OpenFileName-'], preview_area)
            img_arr = aux.ImageArray(image, values['-IsCylindricalMap-'], values['-OblatenessInput-'])

            window.start_thread(
                lambda: aux.image_parser(preview,
                    preview_flag=True,
                    oblateness=aux.float_parser(values['-OblatenessInput-']),
                    latitude_system=values['-LatitudeSystemInput-'],
                    shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                    color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                    albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                    sRGB_gamma=values['-srgbGammaCheckbox-'],
                    #custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
                    maximize_brightness=values['-MaximizeBrightnessCheckbox-'],
                    grid=values['-GridCheckbox-'],
                    log=logger
                )
            )
            window['-InputPreview-'].update(data=aux.img2bytes(preview))

        elif event == '-SaveFileName-':
            if image is not None:
                window.start_thread(
                    lambda: aux.image_parser(image,
                        save_file=values['-SaveFileName-'],
                        oblateness=aux.float_parser(values['-OblatenessInput-']),
                        latitude_system=values['-LatitudeSystemInput-'],
                        shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                        color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                        albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                        sRGB_gamma=values['-srgbGammaCheckbox-'],
                        #custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
                        maximize_brightness=values['-MaximizeBrightnessCheckbox-'],
                        grid=values['-GridCheckbox-'],
                        log=logger
                    )
                )

        # Getting messages from image processing thread
        elif event[0] == '-Thread-':
            print(event[1])
            if values[event] is not None:
                window['-OutputPreview-'].update(data=aux.img2bytes(values[event]))

        # Preview processing
        elif preview is not None:
            window.start_thread(
                lambda: aux.image_parser(preview,
                    preview_flag=True,
                    oblateness=aux.float_parser(values['-OblatenessInput-']),
                    latitude_system=values['-LatitudeSystemInput-'],
                    shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                    color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                    albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                    sRGB_gamma=values['-srgbGammaCheckbox-'],
                    #custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
                    maximize_brightness=values['-MaximizeBrightnessCheckbox-'],
                    grid=values['-GridCheckbox-'],
                    log=logger
                )
            )

        if event in mean_brightness_triggers:
            window['-InputRGB-'].update(img_arr.formatted_mean_brightness())

    window.close()
