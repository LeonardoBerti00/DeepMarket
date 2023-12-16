import constants as cst

HP_CSDI = {
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: {'values': [0.0, 0.1]},
    cst.LearningHyperParameter.DROPOUT.value: {'values': [0, 0.1]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.0003, 0.003, 0.001]},
    cst.LearningHyperParameter.CSDI_LAYERS.value: {'values': [2, 4, 6, 8]},
}

HP_CSDI_FIXED = {
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: 0.1,
    cst.LearningHyperParameter.DROPOUT.value: 0.1,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.CSDI_LAYERS: 1,
}