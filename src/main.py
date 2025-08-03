import FreeSimpleGUI as sg

import src.gui as gui
import src.auxiliary as aux


def launch_window():

    # GUI configuration
    preview_size = (256, 128)
    preview_area = preview_size[0]*preview_size[1]
    available_projections = tuple(aux.projections_dict.keys())

    # Loading the icon
    with open('src/images/icon', 'rb') as file:
        icon = file.read()

    # Launching window
    sg.theme('MaterialDark')
    window = sg.Window(
        'Cylindrical Texture Calibrator', icon=icon, finalize=True, resizable=True,
        layout=gui.generate_layout(preview_size, available_projections)
    )

    logger = gui.create_logger(window, '-Thread-')

    image = preview = None

    # Window events loop
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        elif event == '-OpenFileName-':
            image, preview = aux.image_reader(values['-OpenFileName-'], preview_area)
            window.start_thread(
                lambda: aux.image_parser(preview,
                    preview_flag=True,
                    oblateness=aux.float_parser(values['-OblatenessInput-']),
                    projection=values['-ProjectionInput-'] if values['-ReprojectCheckbox-'] else '',
                    shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                    color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                    albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                    sRGB_gamma=values['-srgbGammaCheckbox-'],
                    custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
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
                        projection=values['-ProjectionInput-'] if values['-ReprojectCheckbox-'] else '',
                        shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                        color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                        albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                        sRGB_gamma=values['-srgbGammaCheckbox-'],
                        custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
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
                    projection=values['-ProjectionInput-'] if values['-ReprojectCheckbox-'] else '',
                    shift=aux.float_parser(values['-ShiftInput-']) if values['-ShiftCheckbox-'] else None,
                    color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                    albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                    sRGB_gamma=values['-srgbGammaCheckbox-'],
                    custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
                    maximize_brightness=values['-MaximizeBrightnessCheckbox-'],
                    grid=values['-GridCheckbox-'],
                    log=logger
                )
            )

    window.close()
