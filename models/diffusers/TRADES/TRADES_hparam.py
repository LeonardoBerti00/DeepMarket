import constants as cst

HP_TRADES = {
    cst.LearningHyperParameter.TRADES_DEPTH.value: {'values': [4, 8]},
    cst.LearningHyperParameter.AUGMENT_DIM.value: {'values': [256]},
    cst.LearningHyperParameter.SEQ_SIZE.value: {'values': [256]},
}


HP_TRADES_FIXED = {
    cst.LearningHyperParameter.TRADES_DEPTH.value: 4,
    cst.LearningHyperParameter.AUGMENT_DIM.value: 64,
    cst.LearningHyperParameter.SEQ_SIZE.value: 256,
} 