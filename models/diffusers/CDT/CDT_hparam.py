import constants as cst

HP_CDT = {
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [128, 256]},
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: {'values': [0.0, 0.1, 0.2]},
    cst.LearningHyperParameter.DROPOUT.value: {'values': [0, 0.1]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.0003, 0.003, 0.001]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value, cst.Optimizers.LION.value]},
    cst.LearningHyperParameter.CDT_DEPTH.value: {'values': [2, 4, 6, 8, 12]},
    cst.LearningHyperParameter.CDT_NUM_HEADS.value: {'values': [4, 8]},
}

HP_CDT_FIXED = {
    cst.LearningHyperParameter.BATCH_SIZE.value: 256,
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: 0.1,
    cst.LearningHyperParameter.DROPOUT.value: 0.1,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.1,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.CDT_DEPTH.value: 6,
    cst.LearningHyperParameter.CDT_NUM_HEADS.value: 4,
}