#! /usr/bin/env python
import csv
import itertools
from collections import defaultdict

with open("airbnb.csv", newline="") as fid:
    reader = csv.DictReader(
        itertools.islice(fid, 1, None),
        delimiter=";",
        fieldnames=(
            "ID",
            "Listing Url",
            "Scrape ID",
            "Last Scraped",
            "Name",
            "Summary",
            "Space",
            "Description",
            "Experiences Offered",
            "Neighborhood Overview",
            "Notes",
            "Transit",
            "Access",
            "Interaction",
            "House Rules",
            "Thumbnail Url",
            "Medium Url",
            "Picture Url",
            "XL Picture Url",
            "Host ID",
            "Host URL",
            "Host Name",
            "Host Since",
            "Host Location",
            "Host About",
            "Host Response Time",
            "Host Response Rate",
            "Host Acceptance Rate",
            "Host Thumbnail Url",
            "Host Picture Url",
            "Host Neighbourhood",
            "Host Listings Count",
            "Host Total Listings Count",
            "Host Verifications",
            "Street",
            "Neighbourhood",
            "Neighbourhood Cleansed",
            "Neighbourhood Group Cleansed",
            "City",
            "State",
            "Zipcode",
            "Market",
            "Smart Location",
            "Country Code",
            "Country",
            "Latitude",
            "Longitude",
            "Property Type",
            "Room Type",
            "Accommodates",
            "Bathrooms",
            "Bedrooms",
            "Beds",
            "Bed Type",
            "Amenities",
            "Square Feet",
            "Price",
            "Weekly Price",
            "Monthly Price",
            "Security Deposit",
            "Cleaning Fee",
            "Guests Included",
            "Extra People",
            "Minimum Nights",
            "Maximum Nights",
            "Calendar Updated",
            "Has Availability",
            "Availability 30",
            "Availability 60",
            "Availability 90",
            "Availability 365",
            "Calendar last Scraped",
            "Number of Reviews",
            "First Review",
            "Last Review",
            "Review Scores Rating",
            "Review Scores Accuracy",
            "Review Scores Cleanliness",
            "Review Scores Checkin",
            "Review Scores Communication",
            "Review Scores Location",
            "Review Scores Value",
            "License",
            "Jurisdiction Names",
            "Cancellation Policy",
            "Calculated host listings count",
            "Reviews per Month",
            "Geolocation",
            "Features",
        ),
    )

    listings = defaultdict(list)

    # import ipdb

    # ipdb.set_trace()
    for row in reader:
        listings[row["Country"]].append(row["ID"])

for country, ids in listings.items():
    with open(f"{country}.txt", "w") as fid:
        fid.write("\n".join(ids))