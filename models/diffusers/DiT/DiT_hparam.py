import constants as cst

HP_DiT = {
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [64, 128, 256]},
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: {'values': [0.1, 0.2]},
    cst.LearningHyperParameter.DROPOUT.value: {'values': [0, 0.1]},
    cst.LearningHyperParameter.DIFFUSION_STEPS.value: {'values': [1000, 4000]},
    cst.LearningHyperParameter.AUGMENT_DIM.value: {'values': [32, 64]},
    cst.LearningHyperParameter.MASKED_SEQ_SIZE.value: {'values': [1, 10]},
    cst.LearningHyperParameter.COND_TYPE.value: {'values': ['full', 'only_event']},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'distribution': 'uniform', 'min':0.00001,'max': 0.001},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value, cst.Optimizers.SGD.value, cst.Optimizers.LION.value]},
    cst.LearningHyperParameter.DiT_DEPTH.value: {'values': [2, 4, 6, 8, 12]},
    cst.LearningHyperParameter.DiT_NUM_HEADS.value: {'values': [4, 6, 12]},
    cst.LearningHyperParameter.DiT_TYPE.value: {"adaln_zero", "concatenation", "cross_attention"}
}