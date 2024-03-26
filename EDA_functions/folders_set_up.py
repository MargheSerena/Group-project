import os

def generate_folders():
    '''This function automatically sets up the path for the following folders based on where you are currently working.
    The function assumes that you have set up a main folder where the following folders are set up:
        - github_folder: where the notebook from which you are calling the function is stored
        - input_folder: where input files are saved. This should be called '01 Input'
        - output folder: where analysis outputs are saved. This should be called '02 Output'
    The outputs of this function (in order) are: github_folder, input_folder, output_folder
    '''
    github_folder = os.getcwd()
    main_folder = os.path.dirname(github_folder)
    
    input_folder = os.path.join(main_folder, "01 Input")
    output_folder = os.path.join(main_folder, "02 Output")

    return github_folder, input_folder, output_folder