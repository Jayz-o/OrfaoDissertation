# NB: 
1. Augmentation before and after the split is done for future work, along with using the HSV colour space, and training GANs for bona fide samples.
2. The Results for the dissertation are available at:.
# References
* This project utilises NVIDIA's pytorch implementation of StyleGan3: https://github.com/NVlabs/stylegan3
* This project was used as a guide: https://www.kaggle.com/raufmomin/vision-transformer-vit-fine-tuning/notebook
# Requirements
1. Create a python 3.8 virtual environment
2. Install FAISS: conda install -c pytorch faiss-gpu
3. Install the requirements.txt: pip install -r requirements.txt
4. Use the 'vgg_face_corrector.py' script to replace 'keras.engine.topology' with 'tensorflow.keras.utils' in <installation path>/keras_vggface/models.py </br>
# Notes
1. On ubuntu, running the script in pycharm might not detect any GPUs. If this happens, launch pycharm from the terminal.
E.g. command: pycharm-professional.
2. **Before running any script in a terminal**, you need to export the path to the project directory. 
E.g. enter the following command in the terminal and then run the script: </br> 
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/root" </br>
On my system: </br> export PYTHONPATH="${PYTHONPATH}:/home/jarred/Documents/OrfaoDissertation"
3. Adjusted StyleGAN3 scripts to fix import errors:
   * Used pickle_helper to fix module paths. Used in metric_utils line 56
   * renamed torch_utils on line 226 of persistence.py.
4. added the following to ray/tune/resources.py line 55:
    *    if type(cpu) is tuple:
            cpu = cpu[0]
         if type(gpu) is tuple:
            gpu = gpu[0]
# Usage
## Create Datasets
### Create CASIA Dataset
1. Go to the ray_casia_dataset_creator.py script
2. Set the CASIA-FA dataset root directory
3. Set the directory to save the processed dataset to
4. Set the number of GPUs and CPUs to use with each trial. Note to run 5 trials on your '1' GPU, set the GPUs to '0.2'.
 This is because 1.0 / 0.2 = 5 trials that will use the GPU. Similarly, set your CPUs.
5. Run the script.

### Create Spoof in the Wild Dataset
1. Go to the ray_siw_dataset_creator.py script
2. Set the SiW dataset root directory
3. Set the directory to save the processed dataset to
4. Set the number of GPUs and CPUs to use with each trial. Note to run 5 trials on your '1' GPU, set the GPUs to '0.2'.
 This is because 1.0 / 0.2 = 5 trials that will use the GPU. Similarly, set your CPUs.
5. Run the script.
