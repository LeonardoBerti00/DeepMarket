import constants as cst

HP_CDT = {
    cst.LearningHyperParameter.CDT_DEPTH.value: {'values': [4, 16]},
    cst.LearningHyperParameter.AUGMENT_DIM.value: {'values': [64, 256]},
    cst.LearningHyperParameter.SEQ_SIZE.value: {'values': [64, 256]},
}


HP_CDT_FIXED = {
    cst.LearningHyperParameter.CDT_DEPTH.value: 8,
    cst.LearningHyperParameter.AUGMENT_DIM.value: 128,
    cst.LearningHyperParameter.SEQ_SIZE.value: 256,
} 