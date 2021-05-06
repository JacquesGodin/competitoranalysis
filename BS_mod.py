from bs4 import BeautifulSoup
import os
import spacy
import argparse
import keyword_extractor

def proc(output):
        
    print(output,"\n\n")
    #####Load the html file to be analysed
    soup = BeautifulSoup((open(output, encoding="utf-8").read()), features="html.parser") #<------------------------- PODE NÃO ESTAR A APANHAR TUDO

    #Collect the name of the competitor under analysis
    ent_name = output.split("_")[-1].split('.')[0]
    new_folder = os.path.join('C:\\Users\\TiagoGodinho\\Desktop\\Projects\\scraper.cfg\\',ent_name)

    text = str(soup.getText(separator="\n", strip=True))
    
    #Create a file under the competitor's name and save the text processed by BeautifulSoup
    file = os.path.join(new_folder, f"{ent_name}.txt")
    
    
    text_file = open(file, "w", encoding="utf-8")
    n = text_file.write(text)
    text_file.close()

    print(text_file.name)
    
    keyword_extractor.analysis(text_file.name)

if __name__ == "__main__":
    #Parser for the output html file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("htmlf", help="output html to be analyzed",type=str, nargs='?', default="scrape_output_loqr.html")

    args = parser.parse_args()


    # transform the parsed argument in the variable for the scraper
    output = args.htmlf

    proc(output)