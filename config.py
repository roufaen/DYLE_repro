class Config:
    # learning rate
    max_grad_norm = 1.0
    retriever_start_lr = 5e-5
    generator_start_lr = 5e-5
    max_decay_num = 3
    no_improvement_decay = 2

    # model
    max_retrieval_len = 512
    max_chunks = 50
    top_k = 6
    max_source_len = 100
    max_target_len = 200
    retriever_alpha = 1
    generation_alpha = 0.5
    consistency_alpha = 1

    # training
    # train_size = 100
    # dev_size = 50
    train_size = 1e9
    dev_size = 1e9
    gradient_accumulation_steps = 8
    save_steps = 20000
    train_log_steps = 10000
    validation_log_steps = 1000

    # directory
    base_dir = '/data2/private/luofuwen/'
    output_dir = base_dir + 'DYLE_repro/outputs/'
    save_model_dir = base_dir + 'DYLE_repro/saved_models/'
    model_path = base_dir + 'models/cpm1-small/'
    data_path = base_dir + 'datasets/CNewSum/'
