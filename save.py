import os
import argparse

# transform the parsed argument in the variable for the scraper

def saver(folder_name):
    folder = os.path.join('C:\\Users\\TiagoGodinho\\Desktop\\Projects\\scraper.cfg\\', folder_name)
    os.makedirs(folder)
    print(f"\n\n\nNew folder created with {folder_name} name\n\n\n")
    return folder
    print(folder, "\n\n\n")
    print(type(folder), "\n\n\n")

if __name__ == "__main__":
    #Parser for the output html file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name of the folder",type=str)

    args = parser.parse_args()

    # transform the parsed argument in the variable for the scraper
    folder_name = args.name

    saver(folder_name)