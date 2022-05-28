### Parameters

* Orignal GAN
  ```
  ./pathgan> python train.py -h
  usage: top [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--g_lr G_LR] [--md_lr MD_LR] [--pd_lr PD_LR]
             [--load_dir LOAD_DIR] [--save_dir SAVE_DIR]

  Training GAN from original paper (https://arxiv.org/pdf/2012.03490.pdf)

  optional arguments:
    -h, --help            show this help message and exit
    --batch_size BATCH_SIZE
                          "Batch size" with which GAN will be trained (default: 8)
    --epochs EPOCHS       Number of "epochs" GAN will be trained (default: 3)
    --g_lr G_LR           "Learning rate" of Generator (default: 0.0001)
    --md_lr MD_LR         "Learning rate" of Map Discriminator (default: 0.00005)
    --pd_lr PD_LR         "Learning rate" of Point Discriminator (default: 0.00005)
    --load_dir LOAD_DIR   Load directory to continue training (default: "None")
    --save_dir SAVE_DIR   Save directory (default: "gan/checkpoint/generator.pth")
  ```
* Pix2Pix GAN
  ```
  ./pathgan> python train_pix2pix.py -h
  usage: top [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--g_lr G_LR] [--d_lr D_LR] [--load_dir LOAD_DIR]
             [--save_dir SAVE_DIR]

  Training Pix2Pix GAN (our GAN)

  optional arguments:
    -h, --help            show this help message and exit
    --batch_size BATCH_SIZE
                          "Batch size" with which GAN will be trained (default: 8)
    --epochs EPOCHS       Number of "epochs" GAN will be trained (default: 20)
    --g_lr G_LR           "Learning rate" of Generator (default: 0.001)
    --d_lr D_LR           "Learning rate" of Discriminator (default: 0.0007)
    --load_dir LOAD_DIR   Load directory to continue training (default: "None")
    --save_dir SAVE_DIR   Save directory (default: "gan/checkpoint/pixgenerator.pth")
  ```
