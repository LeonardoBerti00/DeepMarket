import constants as cst

HP_CDT = {
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: {'values': [0.0]},
    cst.LearningHyperParameter.CDT_DEPTH.value: {'values': [4, 8, 12]},
    cst.LearningHyperParameter.CDT_NUM_HEADS.value: {'values': [1, 2]},
}


HP_CDT_FIXED = {
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: 0.0,
    cst.LearningHyperParameter.CDT_DEPTH.value: 8,
    cst.LearningHyperParameter.CDT_NUM_HEADS.value: 1,
    cst.LearningHyperParameter.AUGMENT_DIM.value: 32,
}