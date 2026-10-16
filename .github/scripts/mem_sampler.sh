#!/bin/sh
# Log memory use every 10 s while the tests run (tests.yml). Needs GNU free
# and procps ps: Linux only.
while true; do
  echo "[mem] $(date +%T) $(free -m | awk '/Mem/{print "used " $3 " avail " $7}; /Swap/{print "swap " $3}' | tr '\n' ' ') $(ps -eo rss,args --sort=-rss | awk 'NR>1&&NR<6{printf "%dMB ", $1/1024}')"
  sleep 10
done
