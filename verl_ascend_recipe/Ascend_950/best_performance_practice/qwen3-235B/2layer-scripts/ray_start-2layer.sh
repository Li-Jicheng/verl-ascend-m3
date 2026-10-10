pkill -9 python
pkill -9 VLLM
pkill -9 ray
ray stop --force

export HCCL_DISABLE_NHR=1

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0
export PYTHONUNBUFFERED=1
export MULTI_STREAM_MEMORY_REUSE=1 
export TASK_QUEUE_ENABLE=2
export HCCL_IF_BASE_PORT=24703
export LCAL_COMM_ID=127.0.0.1:27001

export ASCEND_LAUNCH_BLOCKING=0
export CPU_AFFINITY_CONF=1

export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 使用添加了rope优化的Megatron和Mindspeed
export PYTHONPATH=$PYTHONPATH:/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/MindSpeed

# 日志保存
jiaoben="qwen3-235b-2layer.sh"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=/log/256B-1layer-8k/$TIMESTAMP
mkdir -p $LOG_DIR
LOG_PATH=$LOG_DIR/test.log
cp ${jiaoben} ${LOG_DIR}
export ASCEND_PROCESS_LOG_PATH=$LOG_DIR/ascendlog



ulimit -n 32768
# 保证用户基础环境umask设置，避免用户本身环境umask设置导致生成过高权限的文件夹
umask 0027


NNODES=2
NPUS_PER_NODE=8
#修改为对应主节点IP
MASTER_ADDR=""


#修改为当前节点的通信网卡
export SOCKET_IFNAME=""
export GLOO_SOCKET_IFNAME=""
export HCCL_IFNAME=""



#获取当前节点IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          # echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash ${jiaoben} 2>&1 | tee $LOG_PATH
          break
      else
          # echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
