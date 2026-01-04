import FreeSimpleGUI as sg

import src.gui as gui
import src.auxiliary as aux
from PIL import Image
import numpy as np


def launch_window():

    # GUI configuration
    preview_size = (256, 128)
    preview_area = preview_size[0]*preview_size[1]
    is_gamma_corrected = False
    is_cylindrical_map = True
    latitude_systems = tuple(aux.latitude_systems_dict.keys())
    default_latitude_system = latitude_systems[0]
    is_latitude_converted = False
    oblateness = 0

    # Loading the icon
    with open('src/images/icon', 'rb') as file:
        icon = file.read()

    # Launching window
    sg.theme('MaterialDark')
    window = sg.Window(
        'Cylindrical Texture Calibrator', icon=icon, finalize=True, resizable=True,
        layout=gui.generate_layout(
            preview_size, is_gamma_corrected, latitude_systems, default_latitude_system,
            is_cylindrical_map, is_latitude_converted, oblateness
        )
    )

    # Console logger for the parallel processing thread
    logger = gui.create_logger(window, '-Thread-')

    # Creating the default working object
    black_rectangle = Image.fromarray(np.zeros(preview_size, dtype='uint8'))
    image = black_rectangle
    preview = black_rectangle

    # List of events when the preview should not be updated
    blacklist = ('-SaveFileName-',)

    # Window events loop
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        # The checkbox causes updates for GUI elements
        # LatitudeSystem is used only if the input is a cylindrical map
        # Oblateness is used only if the latitude system needs conversion to planetographic
        elif event in ('-IsCylindricalMap-', '-LatitudeSystemInput-'):
            is_cylindrical_map = values['-IsCylindricalMap-']
            is_latitude_converted = is_cylindrical_map and (values['-LatitudeSystemInput-'] is not latitude_systems[0])
            window['-LatitudeSystemText-'].update(text_color=gui.text_colors[not is_cylindrical_map])
            window['-OblatenessText-'].update(text_color=gui.text_colors[not is_latitude_converted])
            window['-OblatenessInput-'].update(disabled=not is_latitude_converted)
            window['-PlanetographicLatitudesText-'].update(gui.planetographic_latitudes_texts[is_latitude_converted])

        # Opening a file starts processing thread
        elif event == '-OpenFileName-':
            image, preview = aux.image_reader(values['-OpenFileName-'], preview_area)
            window['-InputPreview-'].update(data=aux.img2bytes(preview))

        elif event == '-SaveFileName-':
            image_arr = aux.ImageArray(
                image,
                values['-IsGammaCorrected-'],
                values['-IsCylindricalMap-'],
                values['-LatitudeSystemInput-'],
                aux.float_parser(values['-OblatenessInput-'])
            )
            if image is not None:
                window.start_thread(
                    lambda: aux.image_parser(
                        image_arr,
                        save_file=values['-SaveFileName-'],
                        shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                        color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                        albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                        sRGB_gamma=values['-srgbGammaCheckbox-'],
                        maximize_brightness=values['-MaximizeBrightnessCheckbox-'],
                        grid=values['-GridCheckbox-'],
                        log=logger
                    )
                )

        # Getting messages from image processing thread
        if event[0] == '-Thread-':
            print(event[1])
            if values[event] is not None:
                output_preview_arr = values[event]
                window['-OutputRGB-'].update(output_preview_arr.formatted_mean_brightness())
                window['-OutputPreview-'].update(data=output_preview_arr.to_bytes())

        # Preview processing
        elif event not in blacklist:
            preview_arr = aux.ImageArray(
                preview,
                values['-IsGammaCorrected-'],
                values['-IsCylindricalMap-'],
                values['-LatitudeSystemInput-'],
                aux.float_parser(values['-OblatenessInput-'])
            )
            window['-InputRGB-'].update(preview_arr.formatted_mean_brightness())
            window.start_thread(
                lambda: aux.image_parser(
                    preview_arr,
                    preview_flag=True,
                    shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                    color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                    albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                    sRGB_gamma=values['-srgbGammaCheckbox-'],
                    maximize_brightness=values['-MaximizeBrightnessCheckbox-'],
                    grid=values['-GridCheckbox-'],
                    log=logger
                )
            )

    window.close()
