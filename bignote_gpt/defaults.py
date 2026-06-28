# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

class Defaults:
    config_path         = "dev/train.toml"
    seed                = None
    device              = "cpu"
    deterministic_cuda  = False
    cuda_compile        = False
    tensorboard         = False
    tensorboard_log_dir = None
    corpus_path         = "dev/corpus.txt"
    training_weight     = 80
    validation_weight   = 10
    testing_weight      = 10
    tokenizer_type      = "byte_level"
    model_type          = "classic_gpt"
    optimizer_type      = "adamw"
    model_path          = "dev/model.pt"
    batch_size          = 64
    iteration_limit     = 10000
    warmup_count        = 100
    learning_rate       = 6e-4
    min_learning_rate   = 6e-5
    evaluation_period   = 250
    evaluation_count    = 200
    patience            = 4
    gradient_clip       = 1.0
    prompt              = "\n"
    stop_at             = "\n"
    stop_count          = 4
    token_count         = 1000
    temp                = 0.85
    top_k               = 64
    class help:
        config_path         = "Path to a training configuration file."
        seed                = "Random seed for reproducibility."
        device              = "Device to train on: \"cpu\" or \"cuda\"."
        deterministic_cuda  = "Force deterministic CUDA kernels if seed is set."
        cuda_compile        = "Compile the model if device is \"cuda\"."
        tensorboard         = "Log training metrics to TensorBoard."
        tensorboard_log_dir = "Directory for TensorBoard logs. (./runs if not set)"
        corpus_path         = "Path to a corpus to train on."
        training_weight     = "Relative size of the training split."
        validation_weight   = "Relative size of the validation split."
        testing_weight      = "Relative size of the test split."
        tokenizer_type      = "Tokenizer to use."
        model_type          = "Training model."
        optimizer_type      = "Optimizer to use."
        model_path          = "Path to a trained model file."
        batch_size          = "Training batch size."
        iteration_limit     = "Iteration limit to stop at."
        warmup_count        = "Iterations to linearly warm up the learning rate."
        learning_rate       = "Peak learning rate (after warmup)."
        min_learning_rate   = "Final learning rate at the end of cosine decay."
        evaluation_period   = "Evaluate every N iterations."
        evaluation_count    = "Batches per evaluation."
        patience            = "Stop after N evals without improvement (0 = disabled)."
        gradient_clip       = "Max gradient norm for clipping."
        prompt              = "Seed prompt."
        stop_at             = "Output sequence to stop on."
        stop_count          = "Number of stop sequences before stopping."
        token_count         = "Maximum number of tokens to generate."
        temp                = "Sampling temperature."
        top_k               = "Restrict sampling to the top-k tokens."
