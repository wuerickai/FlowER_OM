import os

DATA_NAME = os.environ.get("DATA_NAME", "USPTO")
EXP_NAME = os.environ.get("EXP_NAME", "")

SCALE = int(os.environ.get("SCALE", 4)) # train & val
# SCALE = 1 # test
SAMPLE_SIZE = 64 // SCALE
NUM_GPU = int(os.environ.get("NUM_GPUS_PER_NODE", 1))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 4096))
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE", 1))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", 512 * NUM_GPU * SCALE))

NUM_NODES = int(os.environ.get("NUM_NODES", 1))
ACCUMULATION_COUNT = int(os.environ.get("ACCUMULATION_COUNT", 1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))

MODEL_NAME = os.environ.get("MODEL_NAME")

def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

class Args:
    # train #
    model_name = MODEL_NAME
    exp_name = EXP_NAME
    oxd_states = False
    pred_del_BE = False
    diverse_flow = False
    reverse_flow = False
    reverse_aug = False
    train_path = os.environ.get("TRAIN_FILE") 
    val_path = os.environ.get("VAL_FILE")
    test_path = os.environ.get("TEST_FILE")
    model_path = os.environ.get("MODEL_PATH")
    result_path = os.environ.get("RESULT_PATH")
    beam_success = os.environ.get("BEAM_SUCCESS")
    data_name = f"{DATA_NAME}"
    log_file = f"FlowER"
    load_from = ""
    seed = int(os.environ.get("SEED", 42))
    resume = True
    load_from = f"{model_path}{MODEL_NAME}"

    backend = "nccl"
    num_workers = NUM_WORKERS
    emb_dim = int(os.environ.get("EMB_DIM"))
    enc_num_layers = 12
    post_processing_layers = 1
    enc_heads = 32
    enc_filter_size = 2048
    dropout = 0.0
    attn_dropout = 0.0
    rel_pos = "emb_only"
    shared_attention_layer = 0
    sigma = float(os.environ.get("SIGMA"))
    train_batch_size = (TRAIN_BATCH_SIZE / ACCUMULATION_COUNT / NUM_GPU / NUM_NODES)
    val_batch_size = (VAL_BATCH_SIZE / ACCUMULATION_COUNT / NUM_GPU / NUM_NODES)
    test_batch_size = TEST_BATCH_SIZE
    batch_type = "tokens_sum"
    lr = 0.0001
    beta1 = 0.9
    beta2 = 0.998
    eps = 1e-9
    weight_decay = 1e-2
    warmup_steps = 30000
    clip_norm = 200

    epoch = int(os.environ.get("EPOCH", 200))
    max_steps = 4000000
    accumulation_count = ACCUMULATION_COUNT
    save_iter = int(os.environ.get("SAVE_ITER", 30000))
    log_iter = int(os.environ.get("LOG_ITER", 200))
    eval_iter = int(os.environ.get("EVAL_ITER", 30000))

    sample_size = SAMPLE_SIZE
    rbf_low = 0
    rbf_high = float(os.environ.get("RBF_HIGH"))
    rbf_oxd_low = float(os.environ.get("RBF_OXD_LOW"))
    rbf_oxd_high = float(os.environ.get("RBF_OXD_HIGH"))
    rbf_oxd_gap = float(os.environ.get("RBF_OXD_GAP"))
    rbf_gap = float(os.environ.get("RBF_GAP"))
    rbf_delBE_low =  float(os.environ.get("RBF_BE_LOW"))
    rbf_delBE_high = float(os.environ.get("RBF_BE_HIGH"))
    rbf_delBE_gap = float(os.environ.get("RBF_BE_GAP"))

    inf_tol = 1e-4
    diverse_lambda = 1

    # validation #
    # do_validate = True
    # steps2validate =  ["1050000"]
    # Set VALIDATION_TOP1=1 to report top-1 alongside the existing top-k metric.
    validation_top1 = _env_flag("VALIDATION_TOP1")

    # # inference # 
    do_validate = False

    # beam-search #
    beam_size = 5
    nbest = 5
    max_depth = 10
    chunk_size = 20
