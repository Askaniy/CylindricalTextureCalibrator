import FreeSimpleGUI as sg

import src.gui as gui
import src.auxiliary as aux


def launch_window():

    # GUI configuration
    preview_size = (256, 128)
    preview_area = preview_size[0]*preview_size[1]

    # Launching window
    sg.ChangeLookAndFeel('MaterialDark')
    window = sg.Window(
        'Cylindrical Texture Calibrator', finalize=True, resizable=True,
        layout=gui.generate_layout(preview_size)
    )

    logger = gui.create_logger(window, '-Thread-')

    image = preview = None

    # Window events loop
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        elif event == '-OpenPath-':
            image, preview = aux.image_reader(values['-OpenPath-'], preview_area)
            window['-InputPreview-'].update(data=aux.convert_to_bytes(preview))
            window.start_thread(
                lambda: aux.image_parser(preview,
                    preview_flag=True,
                    oblateness=aux.float_parser(values['-OblatenessInput-']),
                    color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                    albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                    sRGB_gamma=values['-srgbGammaCheckbox-'],
                    custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
                    log=logger
                ),
                ('-Thread-', 'End of the first preview processing thread\n')
            )
        
        elif event == '-SaveFolder-':
            if image is not None:
                window.start_thread(
                    lambda: aux.image_parser(image,
                        save_folder=values['-SaveFolder-'],
                        oblateness=aux.float_parser(values['-OblatenessInput-']),
                        color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                        albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                        sRGB_gamma=values['-srgbGammaCheckbox-'],
                        custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
                        log=logger
                    ),
                    ('-Thread-', 'End of the image processing thread\n')
                )
        
        # Getting messages from image processing thread
        elif event[0] == '-Thread-':
            print(event[1])
            if values[event] is not None:
                window['-OutputPreview-'].update(data=aux.convert_to_bytes(values[event]))
        
        # Preview processing
        elif preview is not None:
            window.start_thread(
                lambda: aux.image_parser(preview,
                    preview_flag=True,
                    oblateness=aux.float_parser(values['-OblatenessInput-']),
                    color_target=aux.color_parser(values['-ColorInput-']) if values['-ColorCheckbox-'] else None,
                    albedo_target=aux.float_parser(values['-AlbedoInput-']) if values['-AlbedoCheckbox-'] else None,
                    sRGB_gamma=values['-srgbGammaCheckbox-'],
                    custom_gamma=aux.float_parser(values['-GammaInput-']) if values['-CustomGammaCheckbox-'] else None,
                    log=logger
                ),
                ('-Thread-', 'End of the preview processing thread\n')
            )

    window.close()