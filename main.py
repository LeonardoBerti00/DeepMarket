from run import run
import torch



if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    run()