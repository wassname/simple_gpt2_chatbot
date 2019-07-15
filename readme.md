Experiment to see if I can make a chatbot in the style of reddit.com/u/disumbrationist's SubSimulatorGPT2. 

- First download data, or get it from here http://publicmldatasets.thinkcds.com/transfer-learning-conv-ai/20190715_reddit_threads_pickle.tar.gz
- Process to text files with `A_Format_data_for_simple_gpt2.ipynb`
- Run colab notebook to train simple gpt2 medium on a TPU `B_COLAB_Train_on_reddit_machine_learning_GPT_2_Text_Generating_Model_w_GPU.ipynb`
- Download the pytorch version of the checkpoint (see the notebook). And load it in `C_try_running_gpt2_simple.ipynb`
