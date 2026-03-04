# 2026-DATCFTNet-SPEECH ENHANCEMENT FOR COCHLEAR IMPLANT RECIPIENTS USING ATTENTION-BASED DUAL-PATH RECURRENT NEURAL NETWORK

# ******************* CFTNet ******************
 Complex-valued Frequency Transformation Network for Speech Enhancement
 Authors: Nursadul Mamun, John H.L. Hansen “DATCFTNet: DATCFTNet-SPEECH ENHANCEMENT FOR COCHLEAR IMPLANT RECIPIENTS USING ATTENTION-BASED DUAL-PATH RECURRENT NEURAL NETWORK, ICASSP, Spain, Barcelona, 2026.

# Architecture
The DAT-CFTNet model extends the complex-valued frequency transformation network (CFTNet) by incorporating a Dual-Path Attention RNN (DAT-RNN) module in the bottleneck layer to improve speech enhancement for cochlear implant (CI) users. The architecture follows an encoder–bottleneck–decoder structure where noisy speech is first converted into a time–frequency representation using STFT and processed through complex-valued convolutional encoder layers and frequency transformation blocks to capture local and global spectral correlations. In the bottleneck stage, the proposed DAT-RNN replaces the conventional GRU units and combines dual-path recurrent processing with an attention mechanism to model both intra-chunk (local time–frequency segments) and inter-chunk (global sequence dependencies) relationships. The intra-chunk branch uses Bi-LSTM to capture fine spectral–temporal patterns within each segment, while the inter-chunk branch employs LSTM to aggregate information across segments, and an attention module dynamically assigns importance weights to emphasize salient speech features while suppressing noise. The decoder then reconstructs the enhanced speech spectrum using skip connections from the encoder, enabling accurate recovery of speech structure while reducing noise artifacts. This design allows DAT-CFTNet to effectively capture long-range dependencies and improve speech intelligibility and quality, particularly for CI listeners operating under noisy conditions.



![CFTNet_Network_Overview](https://github.com/nursad49/2023-CFTNet-Complex-valued-Frequency-Transformation-Network-for-Speech-Enhancement/assets/45471274/bb8c451b-2a6b-4c9c-b1fb-782be8232157)




The folder contains:

AudioDataGeneration.py: This generates the noisy audio samples for different SNRs for the Train, Dev, and Test folders. It saves audio and noisy audio files 			in the Database>Original_Samples>Train/Dev/Test folder. Change the name of the noise and SNRs in the function.

Write_scp_files.py: This generates the .scp files for Dataprep.py. Change the noise name as like AudioDataGenerator.py. 

Dataprep.py: This splits all clean and noisy audio files into 4-second chunks and generates training samples to use to train the system. The generated files 	    are saved in the Database>Training_Samples folder

Network.py: This file contains all proposed networks (DATCFTNET). You can test each network by running this files separately. 

Train.py: This file contains all files related to train the model. This import training data from Database>Training_Samples folder and related model from the 	Network.py file.

Test.py: This evaluates the network by using the test samples from Database>Original_Samples>Test folder and generate the enhanced files in the same folder.


Dependencies:

dataloader.py: This helps the train.py function to load all training samples (train and Dev) from Database>Training_Samples folder.

modules.py: It contains all the functions required to design the model.

utils.py: It contains all functions required for the whole system.

objective_metrics.py: This contains all objective speech intelligibility and quality metrics, and loss functions.


How to Run:
												
# Part 1: If running for the first time or folders are not available for this database:

Step 1: Execute AudioDataGeneration.py to generate noisy samples corresponding to clean samples for different noise types and SNRs. Ensure you have clean files in Database>Original_Samples>Clean and noise in Database>Original_Samples>Different_Noise folder to generate rthe equired noisy files. Modify AudioDataGeneration.py according to your specifications for noisy samples and make necessary edits. 
   		*If you already have noisy samples for corresponding clean samples, you can skip this step.*

Step 2: Run `Write_scp_files.py` to generate `Train.scp`, `Dev.scp`, and `Test.scp` files. These files will be utilized in the `Dataprep.py` function to segment all training files.
   		*If you already have noisy samples for corresponding clean samples, you can skip this step.*

Step 3: Execute `Dataprep.py` to segment audio files and create the `Database > Training_Samples` folder.
   		*If you already have these files, you can skip this step.*

# Part 2:  Run these steps to train your model every time



Step 4: To train the model run this function in the Python terminal:-

			python3 train.py --model "$model name" --b "$batch size" --e "$Num of epoch" --loss "$Loss Function" --gpu "$GPUs"
			# For Example: python3 train.py --model DATCFTNet --b 8 --e 50 --loss SISDR+FreqLoss --gpu '0 1'
			# Simple Example: python3 train.py --model DATCFTNet


This will save a .ckpt file in the Saved_Models>$Model Name folder.
			Repeat step 4 every time to run the desired model.


Step 5: To test the model run this function in the Python terminal:-

			“python3 test.py”

Please change the model name, and model file from the “Saved_Models>$Model Name” folder in the main file of test.py before testing.


