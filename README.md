* data preprocess
  + cd dataset/
  + python data_preprocess.py  - -min_length n (n为需要剔除的最小长度)
  + 将原始数据合并&拆分为train,dev,test
* train
  + cd src_code
  + bash train.sh
* test
  + bash test.sh
* experiment results
  | model      |   lr | eval_step | train_steps | per_gpu_train_batch_size | eval_loss | test_loss |
  |------------+------+-----------+-------------+--------------------------+-----------+-----------|
  | bert-base  | 1e-5 |       150 |        6000 |                      128 |    0.8189 |    0.8083 |
  | + 对抗扰动  | 1e-5 |       150 |        5500 |                      128 |    0.7791 |    0.7699 |
