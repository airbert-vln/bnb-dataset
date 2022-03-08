"""
This is scrapping a list of places from Wikipedia.
You should adapt this script according to the page you want to scrap
"""
from pathlib import Path
import argtyped
from selenium import webdriver


class Arguments(argtyped.Arguments):
    url: str = "https://en.wikipedia.org/wiki/List_of_the_most_common_U.S._place_names"
    output: Path = "data/cities.txt"


if __name__ == "__main__":
    args = Arguments()
    print(args)

    driver = webdriver.Firefox()
    driver.get(args.url)

    cities = [
        a.text
        for a in driver.find_elements_by_xpath(
            '//table[@class="wikitable sortable jquery-tablesorter"]//td//a'
        )
    ]
    driver.close()

    args.output.parent.mkdir(exist_ok=True, parents=True)

    with open(args.output, "w") as fid:
        json.dump(cities, fid)
