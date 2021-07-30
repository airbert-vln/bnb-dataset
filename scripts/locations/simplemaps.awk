#! /usr/bin/awk -f

BEGIN  { FS = "," } 

FNR > 1 { gsub(/['\/"]/,"",$1) ; gsub(/-/, " ", $1) ; gsub(/['\/"]/,"",$4) ; gsub(/-/, " ", $1); print $1  " " $4 " Canada" }

