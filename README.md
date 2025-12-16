## Download the data: [data](https://huggingface.co/datasets/johnowhitaker/imagenette2-320)

## Guideline to run code:

**1.Download the data in the link above and edit the direct link in dataloader.py file**  

**2. In 'res' folder, create 2 'child' folders which names 'figs_reg' and 'models_reg' (where the trained model is saved)**

**(The folder 'runs' in current document is redundant, you can delete it before running)**  

**3. In the folder's terminal, run 'python main.py' or equivalent code to start training loop** 

**4. If you want to change hyper-parameters like epochs, batch size, etc. Look for them in 'config' file**

**5. After training loop, we have 'test_result.py' in 'test simulator' folder to test the model. The input is an image, and the output we can see Encoder's output and phosphene images of 2 different kinds of simulators.**

**📌 This work is based on the paper "Point-SPV: End-to-End Enhancement of Object Recognition in Simulated Prosthetic Vision Using Synthetic Viewing Points". We sincerely thank the authors for making their source code publicly available.**
