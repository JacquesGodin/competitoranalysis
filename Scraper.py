from scraper_api import ScraperAPIClient
import requests
import argparse
import save
import os

#API key for ScraperAPI account
client = ScraperAPIClient('d25d1e44145a496d29e291923788dfeb')

#Parser for the website argument
parser = argparse.ArgumentParser()
parser.add_argument("site", help="website to be scraped",type=str)

args = parser.parse_args()

# transform the parsed argument in the variable for the scraper
url = args.site
print(args.site)
print(url)

print("Getting information\n\n")

result = client.get(url).text


#Retrive Competitor's name to use in the output file name
if "www" in url:
    domain=url.split("www.")[-1].split(".")[0]  
else:
    domain=url.split("//")[-1].split(".")[0]  

#Call Save script to create a new folder with the name of the analyzed competitor
new_folder = save.saver(domain) 

#Save output html file to created folder
file = os.path.join(new_folder, f"scrape_output_{domain}.html")
text_file = open(file, "w", encoding="utf-8")
n = text_file.write(result)
text_file.close()

#string variable to pass the right file to the NLP script
output = f"scrape_output_{domain}.html"

print(output)
print("\n\nSaved to output file")

