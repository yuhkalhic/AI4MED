export MACA_PATH=/opt/maca
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export PATH=${CUDA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}

export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/workspace/LLaMA-Factory-main/examples/lora_multi_gpu/lib/cublas/src/:${LD_LIBRARY_PATH}
#export MCBLAS_CUSTOMIZED_CONFIG_PATH=/workspace/LLaMA-Factory-main/examples/lora_multi_gpu/lib/llama3-8b/mcblas_customized_config.yaml



export MACA_SMALL_PAGESIZE_ENABLE=1
export PYTORCH_ENABLE_SAME_SAME_RAND_A100=1
export SET_DEVICE_NUMA_PREFERRED=1


export MCCL_P2P_LEVEL=SYS
export MCCL_FAST_WRITE_BACK=1
export MCCL_EARLY_WRITE_BACK=15
export MCCL_NET_GDR_LEVEL=SYS
export MCCL_CROSS_NIC=1
export MCCL_ENABLE_FC=1

#export MHA_USE_BLAS=ON
export MHA_BWD_NO_ATOMIC_F64=1


llamafactory-cli train lora_llama2_13b_8gpu.yaml
