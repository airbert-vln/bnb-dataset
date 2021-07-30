#! /usr/bin/awk -f

BEGIN  { FS = "," } 

FNR > 1 { gsub(/['\/"]/," ",$2) ; print $2  " " $12 } # gsub(/\"|\;/,"",$12) }

