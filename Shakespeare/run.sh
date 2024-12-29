# p_value
p_value=${p_value:-0.8}
num_clients=${num_clients:-20}
drop_type=${drop_type:-"ordered"}

# Storing all configuration parameters in a string. Used for writing parameter values to a log file.
all_parameters_str="p_value num_clients drop_type"

# Creating unique logging folder
folder_prefix="logs/lstm"
folder_num=1
folder_name="${folder_prefix}_${folder_num}"
while [ -d "${folder_name}" ]; do
    ((folder_num++))
    folder_name="${folder_prefix}_${folder_num}"
done
mkdir -p ${folder_name}
echo "Logging to ${folder_name}"

# Outputting configuration parameters used for this run
log_file_config="${folder_name}/config.log"
for param in $all_parameters_str; do
    echo "${param}: ${!param}" >> ${log_file_config} 
done

# Logging time this run started
echo "Run start: $(TZ=America/Toronto date +"%d %m %Y %T")" >> ${log_file_config} 

source ../venv/bin/activate

# Starting clients
for i in $(seq 1 ${num_clients})
do
    log_file_client="${folder_name}/client_${i}.log"
    drop_type=${drop_type} p_value=${p_value} cid=${i} bash run_shake.sh 1 > ${log_file_client} 2>&1 &
done

# Starting server
log_file_server="${folder_name}/server.log"
drop_type=${drop_type} p_value=${p_value} num_clients=${num_clients} bash run_shake.sh 0 2>&1 | tee ${log_file_server}

# Logging time this run ended
echo "Run end: $(TZ=America/Toronto date +"%d %m %Y %T")" >> ${log_file_config} 

# Get fit progress
log_fit_progress="${folder_name}/fit_progress.log"
grep "fit progress" ${log_file_server} > ${log_fit_progress}