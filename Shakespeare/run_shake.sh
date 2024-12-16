#!/bin/bash

echo $1
cid=$(($1-1))
cid=$(($cid * 10))
echo $cid
if [[ $1 -eq 0 ]]
then 
    echo $1
    hostname -I > ip_shake.txt
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip_shake.txt

    python3 server.py --server_address $server:35005 --rounds 50 --min_num_clients 20 --min_sample_size 1 --model Shakespeare_LSTM

else 
    sleep 20
    echo $1
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip_shake.txt

    for i in $(seq 0 19); do
        python3 client.py --server_address $server:35005 --cid $(($cid  + i)) --model Shakespeare_LSTM --device gpu &
    done
    
fi
