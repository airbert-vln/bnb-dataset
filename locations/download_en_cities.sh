# wget https://simplemaps.com/static/data/canada-cities/1.7/basic/simplemaps_canadacities_basicv1.7.zip
# unzip simplemaps_canadacities_basicv1.7.zip
# ./simplemaps.awk canadacities.csv | sort | uniq > bigcities_canada.txt


for country in nz ie gb au; do 
	wget https://simplemaps.com/static/data/country-cities/${country}/${country}.csv
	awk  'BEGIN  { FS = "," } FNR > 1 { gsub(/"/,"",$1) ; gsub(/-/, " ", $1) ; gsub(/"/,"",$4) ; gsub(/-/, " ", $1); print $1  " " $6 " " $4 }' ${country}.csv | sort | uniq > bigcities_${country}.txt
done
