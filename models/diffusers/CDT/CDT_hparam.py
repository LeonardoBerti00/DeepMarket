import constants as cst

HP_CDT = {
    cst.LearningHyperParameter.CDT_DEPTH.value: {'values': [4, 16]},
    cst.LearningHyperParameter.AUGMENT_DIM.value: {'values': [64, 256]},
    cst.LearningHyperParameter.SEQ_SIZE.value: {'values': [64, 256]},
}


HP_CDT_FIXED = {
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: 0.0,
    cst.LearningHyperParameter.CDT_DEPTH.value: 8,
    cst.LearningHyperParameter.CDT_NUM_HEADS.value: 2,
    cst.LearningHyperParameter.AUGMENT_DIM.value: 128,
}