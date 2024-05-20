import constants as cst

HP_CGAN = {
    cst.LearningHyperParameter.SEQ_SIZE.value: {'values': [256]},
}


HP_CGAN_FIXED = {
    cst.LearningHyperParameter.SEQ_SIZE.value: 256,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
} 