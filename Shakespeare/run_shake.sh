#!/bin/bash
p_value=${p_value:-0.8}
num_clients=${num_clients:-20}
cid=${cid:-0}
drop_type=${drop_type:-"dynamic"}

echo "Command arg is $1"

if [[ $1 -eq 0 ]]
then 
    hostname -I > ip_shake.txt
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip_shake.txt

    DROP_TYPE=${drop_type} P_VALUE=${p_value} python3 server.py --server_address $server:35005 --rounds 250 --min_num_clients ${num_clients} --min_sample_size 1 --model Shakespeare_LSTM

else 
    sleep 5
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip_shake.txt

    echo "cid is: ${cid}"
    DROP_TYPE=${drop_type} P_VALUE=${p_value} python3 client.py --server_address $server:35005 --cid $cid --model Shakespeare_LSTM --device gpu &
    
fi
