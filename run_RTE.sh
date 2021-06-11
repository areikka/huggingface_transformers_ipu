export TASK_NAME=rte
# export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"./profiles/sec_try","autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true"}'
export POPLAR_ENGINE_OPTIONS='{"debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true","autoReport.directory":"./profiles/3rd_try","profiler.perExecutionStreamCopyCycles":"true"}'

python run_glue_no_trainer_ipu.py \
  --model_name_or_path bart-base \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_train_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --OffChipStorage \
  --output_dir /tmp/$TASK_NAME 2>&1 | tee ./logs.log;
