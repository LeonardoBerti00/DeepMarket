import constants as cst
#TODO change dit in gaussiandiffusion
HP_DiT = {
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [128, 256]},
    cst.LearningHyperParameter.CONDITIONAL_DROPOUT.value: {'values': [0.0, 0.1, 0.2]},
    cst.LearningHyperParameter.DROPOUT.value: {'values': [0, 0.1]},
    cst.LearningHyperParameter.AUGMENT_DIM.value: {'values': [32, 64]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'distribution': 'uniform', 'min':0.00001,'max': 0.001},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value, cst.Optimizers.SGD.value, cst.Optimizers.LION.value]},
    cst.LearningHyperParameter.DiT_DEPTH.value: {'values': [2, 4, 6, 8, 12]},
    cst.LearningHyperParameter.DiT_NUM_HEADS.value: {'values': [4, 8]},
}